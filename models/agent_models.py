import math

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn

from .simple_conv import SimpleConv
from .utils import calculatePlace


class SimpleMazeModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.hiddenSize = self.model_config.get("hiddenSize", 16)
        self.numLayers = self.model_config.get("numLayers", 4)
        self.inputSize = self.model_config.get("inputSize", 13)
        self.linearHiddenSize = self.hiddenSize * 8
        self.primaryConvModule = SimpleConv(self.hiddenSize)
        self.primaryConvModuleOutSize = ((self.inputSize + 1) // 2 + 1) // 2
        self.prePredictionHead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.primaryConvModuleOutSize**2 * self.hiddenSize * 2,
                self.linearHiddenSize,
            ),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.policy_branch = nn.Linear(self.linearHiddenSize, self.action_space.n)
        self.value_branch = nn.Linear(self.linearHiddenSize, 1)

    def _forward_intermediate(self, batch):
        mapInput = batch[Columns.OBS]
        if type(mapInput) is dict:
            mapInput = mapInput["vision"]
        mapInput = torch.reshape(mapInput, [-1, self.inputSize, self.inputSize, 3])
        mapInput = mapInput.permute(0, 3, 1, 2).to(torch.float32)
        mapOutput = self.primaryConvModule(mapInput)
        return self.prePredictionHead(mapOutput)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        policy = self.policy_branch(self._forward_intermediate(batch))
        return {Columns.ACTION_DIST_INPUTS: policy}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        return self.value_branch(self._forward_intermediate(batch))


class MemoryMazeModule(SimpleMazeModule):
    def setup(self):
        SimpleMazeModule.setup(self)
        self.trajectoryMemory = nn.GRU(
            self.linearHiddenSize, self.linearHiddenSize, batch_first=True
        )

    @override(TorchRLModule)
    def get_initial_state(self):
        return {"hiddenObs": torch.zeros((self.linearHiddenSize,), dtype=torch.float32)}

    def _processConvolution(self, vision) -> torch.Tensor:
        visionShape = vision.shape
        vision = vision.reshape(-1, *visionShape[2:])
        vision = vision.permute(0, 3, 1, 2).to(torch.float32)
        visionFeatures = self.primaryConvModule(vision)
        visionFeatures = self.prePredictionHead(visionFeatures)
        visionFeatures = visionFeatures.reshape(*visionShape[:2], self.linearHiddenSize)
        return visionFeatures

    def _processPreHeads(self, batch):
        initialHidden = batch[Columns.STATE_IN]["hiddenObs"].unsqueeze(0)
        vision = batch[Columns.OBS]
        visionFeatures = self._processConvolution(vision)
        return self.trajectoryMemory(visionFeatures, initialHidden)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        allHiddenStates, finalHiddenState = self._processPreHeads(batch)
        policy = self.policy_branch(allHiddenStates)
        return {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {"hiddenObs": finalHiddenState.squeeze(0)},
            Columns.EMBEDDINGS: allHiddenStates,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        return self.value_branch(self._processPreHeads(batch)[0]).squeeze(-1)


class PathIntegrationWithVisionModule(MemoryMazeModule):
    def setup(self):
        MemoryMazeModule.setup(self)
        self.mazeSize = self.model_config.get("mazeSize", 31)
        self.numPlaceCells = self.model_config.get("numPlaceCells", 32)
        self.gridSize = 512
        self.integratorSize = 128
        self.gridDecoder = nn.Linear(self.integratorSize, self.gridSize)
        self.placeProjector = nn.Sequential(
            nn.Dropout(), nn.Linear(self.gridSize, self.numPlaceCells)
        )
        self.pathIntegrator = nn.LSTM(5, self.integratorSize, batch_first=True)
        self.gridCompressor = nn.Sequential(
            nn.Linear(self.gridSize, self.linearHiddenSize),
            nn.ReLU(),
        )
        self.placeCells = nn.Parameter(torch.rand([self.numPlaceCells, 2]), False)
        self.fieldSize = 0.3 / math.sqrt(self.numPlaceCells)
        self.placeEncoder = nn.Linear(self.numPlaceCells, 2 * self.integratorSize)
        self.placeEncoderForMemory = nn.Linear(
            self.numPlaceCells, self.linearHiddenSize
        )
        self.piPolicyPredictor = nn.Linear(self.linearHiddenSize, self.action_space.n)
        self.piValuePredictor = nn.Linear(self.linearHiddenSize, 1)

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "hiddenObs": torch.zeros((self.linearHiddenSize,), dtype=torch.float32),
            "candidateGrid": torch.zeros((self.integratorSize,), dtype=torch.float32),
            "hiddenGrid": torch.zeros((self.integratorSize,), dtype=torch.float32),
        }

    def _getObsFromBatch(self, batch):
        obs = batch["obs"]
        visionSize = self.inputSize**2 * 2
        vision = obs[:, :, :visionSize]
        vision = torch.reshape(
            vision, [*vision.shape[:2], self.inputSize, self.inputSize, 2]
        )
        lastAgentLocation = obs[:, :, visionSize : visionSize + 2]
        lastAgentLocation = lastAgentLocation.reshape(*lastAgentLocation.shape[:2], 2)
        agentLocation = obs[:, :, visionSize + 2 : visionSize + 4]
        agentLocation = agentLocation.reshape(*agentLocation.shape[:2], 2)
        action = obs[:, :, -5:]
        return vision, lastAgentLocation, agentLocation, action

    def _pathIntegrate(
        self,
        lastAgentLocation: torch.Tensor,
        action: torch.Tensor,
        hiddenGrid: torch.Tensor,
        candidateGrid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prevPlaces = self.placeEncoder(
            calculatePlace(self.placeCells, lastAgentLocation, self.fieldSize)
        )
        actualHiddenGrid = prevPlaces[:, : self.integratorSize].contiguous()
        actualCandidateGrid = prevPlaces[:, self.integratorSize :].contiguous()
        if not self.training:
            hiddenPlace = actualHiddenGrid
            candidatePlace = actualCandidateGrid
            actualHiddenGrid = hiddenGrid
            actualCandidateGrid = candidateGrid
            initialPlaceMask = torch.sum(actualHiddenGrid, 1) == 0
            randomPlaceMask = (
                torch.rand(initialPlaceMask.shape, dtype=torch.float32) < 0.05
            )
            placeMask = torch.where(randomPlaceMask, randomPlaceMask, initialPlaceMask)[
                :, None
            ]
            actualHiddenGrid = torch.where(placeMask, hiddenPlace, actualHiddenGrid)
            actualCandidateGrid = torch.where(
                placeMask, candidatePlace, actualCandidateGrid
            )
        integratedStates, finalIntegrationState = self.pathIntegrator(
            action, (actualHiddenGrid.unsqueeze(0), actualCandidateGrid.unsqueeze(0))
        )
        integratedCode = self.gridDecoder.forward(integratedStates)
        projectedPlace = self.placeProjector(integratedCode)
        return (
            integratedCode.detach(),
            projectedPlace,
            finalIntegrationState,
        )

    def _getInitialMemory(self, lastLocation: torch.Tensor, memoryState: torch.Tensor):
        prevPlaces = self.placeEncoderForMemory(
            calculatePlace(self.placeCells, lastLocation)
        )
        initialPlaceMask = torch.sum(memoryState, 1) == 0
        return torch.where(initialPlaceMask[:, None], prevPlaces, memoryState)

    def _processVisualMemory(self, vision: torch.Tensor, initialHidden: torch.Tensor):
        visionFeatures = self._processConvolution(vision)
        memory, _ = self.trajectoryMemory(visionFeatures, initialHidden.unsqueeze(0))
        return memory

    def _getPolicyAndValue(self, batch):
        memory = None
        value = None
        policy = None
        projectedPlace = None
        finalIntegrationState = None

        vision, lastAgentLocation, _, action = self._getObsFromBatch(batch)

        selfLocalize = self.model_config.get("self_localize", False)
        useVisionPolicy = self.model_config.get("visionPolicy", False)
        if selfLocalize or not useVisionPolicy:
            gridCodes, projectedPlace, finalIntegrationState = self._pathIntegrate(
                lastAgentLocation[:, 0, :],
                action,
                batch[Columns.STATE_IN]["hiddenGrid"],
                batch[Columns.STATE_IN]["candidateGrid"],
            )
            if not selfLocalize:
                integration = self.gridCompressor(gridCodes)
                policy = self.piPolicyPredictor(integration)
                value = self.piValuePredictor(integration)
        elif useVisionPolicy:
            initialState = self._getInitialMemory(
                lastAgentLocation[:, 0, :], batch[Columns.STATE_IN]["hiddenObs"]
            )
            memory = self._processVisualMemory(vision, initialState)
            policy = self.policy_branch(memory)
            value = self.value_branch(memory)

        return (
            policy,
            value,
            memory,
            projectedPlace,
            finalIntegrationState,
        )

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        _, _, agentLocation, _ = self._getObsFromBatch(batch)
        policy, value, visualMemory, projectedPlace, finalGrid = (
            self._getPolicyAndValue(batch)
        )
        obs = batch["obs"]
        if policy is None:
            policy = torch.ones(
                [*obs.shape[:2], self.action_space.n],
                dtype=torch.float32,
                device=obs.device,
            )

        if visualMemory is None:
            visualMemory = torch.randn(
                [*obs.shape[:2], self.linearHiddenSize],
                dtype=torch.float32,
                device=obs.device,
            )

        if value is None:
            value = torch.ones(
                [*obs.shape[:2], 1], dtype=torch.float32, device=obs.device
            )

        output = {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {
                "hiddenObs": visualMemory[:, -1],
                "candidateGrid": finalGrid[1].squeeze(0)
                if finalGrid is not None
                else batch[Columns.STATE_IN]["candidateGrid"],
                "hiddenGrid": finalGrid[0].squeeze(0)
                if finalGrid is not None
                else batch[Columns.STATE_IN]["hiddenGrid"],
            },
            Columns.EMBEDDINGS: value,
        }

        if self.model_config.get("self_localize", False):
            output["placeLogit"] = projectedPlace
            output["placeTarget"] = calculatePlace(
                self.placeCells, agentLocation, self.fieldSize
            )

        return output

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            obs = batch["obs"]
            _, embeddings, _, _, _ = self._getPolicyAndValue(batch)
            if embeddings is None:
                return torch.ones(
                    [*obs.shape[:2]], dtype=torch.float32, device=obs.device
                )
        return embeddings.squeeze(-1)


class PathIntegrationWithVisionModuleForEval(PathIntegrationWithVisionModule):
    def setup(self):
        PathIntegrationWithVisionModule.setup(self)

    def _getPolicyAndValue(self, batch):
        assert not self.training, (
            '"PathIntegrationWithVisionModuleForEval" cannot be used for training'
        )

        vision, lastAgentLocation, _, action = self._getObsFromBatch(batch)
        gridCodes, _, finalIntegrationState = self._pathIntegrate(
            lastAgentLocation[:, 0, :],
            action,
            batch[Columns.STATE_IN]["hiddenGrid"],
            batch[Columns.STATE_IN]["candidateGrid"],
        )

        initialState = self._getInitialMemory(
            lastAgentLocation[:, 0, :], batch[Columns.STATE_IN]["hiddenObs"]
        )
        memory = self._processVisualMemory(vision, initialState)

        integration = self.gridCompressor(gridCodes)
        visualPolicy = self.policy_branch(memory)
        integrationPolicy = self.policy_branch(integration)
        visualConfidence = (
            torch.nn.functional.softmax(visualPolicy, -1)
            .flatten()
            .topk(2)
            .values.sum()
            .item()
        )
        integrationConfidence = (
            torch.nn.functional.softmax(integrationPolicy, -1)
            .flatten()
            .topk(2)
            .values.sum()
            .item()
        )
        if visualConfidence > integrationConfidence:
            policy = visualPolicy
        else:
            policy = integrationPolicy

        return (
            policy,
            None,
            memory,
            None,
            finalIntegrationState,
        )


class GPSModule(MemoryMazeModule):
    def setup(self):
        MemoryMazeModule.setup(self)
        self.trajectoryMemory = nn.GRU(
            self.linearHiddenSize + 2, self.linearHiddenSize, batch_first=True
        )

    def _getObsFromBatch(self, batch):
        obs = batch["obs"]
        visionSize = self.inputSize**2 * 2
        vision = obs[:, :, :visionSize]
        vision = torch.reshape(
            vision, [*vision.shape[:2], self.inputSize, self.inputSize, 2]
        )
        agentLocation = obs[:, :, visionSize + 2 : visionSize + 4]
        agentLocation = agentLocation.reshape(*agentLocation.shape[:2], 2)
        return vision, agentLocation

    def _processPreHeads(self, batch):
        vision, agentLocation = self._getObsFromBatch(batch)
        visionFeatures = self._processConvolution(vision)
        initialHidden = batch[Columns.STATE_IN]["h"].unsqueeze(0)
        visionAndGridFeatures = torch.concat([visionFeatures, agentLocation], dim=2)
        return self.trajectoryMemory(visionAndGridFeatures, initialHidden)
