import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.utils.annotations import override

from .simple_conv import SimpleConv


class SimpleMazeModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.hiddenSize = self.model_config.get("hiddenSize", 16)
        self.numLayers = self.model_config.get("numLayers", 4)
        self.inputSize = self.model_config.get("inputSize", 13)
        self.linearHiddenSize = self.hiddenSize * 8
        self.primaryConvModule = SimpleConv(self.hiddenSize)
        primaryConvModuleOutSize = ((self.inputSize + 1) // 2 + 1) // 2
        self.prePredictionHead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                primaryConvModuleOutSize**2 * self.hiddenSize * 2, self.linearHiddenSize
            ),
            nn.ReLU(),
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
        return {"h": torch.zeros((self.linearHiddenSize,), dtype=torch.float32)}

    def _processConvolution(self, vision):
        visionShape = vision.shape
        vision = vision.reshape(-1, *visionShape[2:])
        vision = vision.permute(0, 3, 1, 2).to(torch.float32)
        visionFeatures = self.primaryConvModule(vision)
        visionFeatures = self.prePredictionHead(visionFeatures)
        visionFeatures = visionFeatures.reshape(*visionShape[:2], self.linearHiddenSize)
        return visionFeatures

    def _processPreHeads(self, batch):
        initialHidden = batch[Columns.STATE_IN]["h"].unsqueeze(0)
        vision = batch[Columns.OBS]
        visionFeatures = self._processConvolution(vision)
        return self.trajectoryMemory(visionFeatures, initialHidden)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        allHiddenStates, finalHiddenState = self._processPreHeads(batch)
        policy = self.policy_branch(allHiddenStates)
        return {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {"h": finalHiddenState.squeeze(0)},
            Columns.EMBEDDINGS: allHiddenStates,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        return self.value_branch(self._processPreHeads(batch)[0]).squeeze(-1)


class PlaceMazeModule(MemoryMazeModule):
    def setup(self):
        SimpleMazeModule.setup(self)
        self.mazeSize = self.model_config.get("mazeSize", 31)
        self.numPlaceCells = (self.mazeSize // 2) ** 2
        self.gridSize = self.linearHiddenSize * 2
        self.gridDecoder = nn.Linear(self.gridSize, self.gridSize)
        self.placeProjector = nn.Sequential(
            nn.Dropout(), nn.Linear(self.gridSize, self.numPlaceCells)
        )
        self.pathIntegrator = nn.LSTM(5, self.gridSize, batch_first=True)
        self.trajectoryMemory = nn.GRU(
            self.linearHiddenSize + self.gridSize,
            self.linearHiddenSize,
            batch_first=True,
        )
        self.initialStates = nn.Embedding(2, self.gridSize)

    def _calculatePlace(self, agentLocation):
        formattedLocation = agentLocation.permute(0, 1, 4, 2, 3).to(torch.float32)
        formattedLocation = agentLocation.reshape([-1, 1, self.mazeSize, self.mazeSize])
        kernel = torch.ones(
            [1, 1, 3, 3], dtype=torch.float32, device=formattedLocation.device
        )
        with torch.no_grad():
            unweightedActivation = torch.nn.functional.conv2d(
                formattedLocation,
                kernel,
                stride=2,
            )
            unweightedActivation = unweightedActivation.flatten(1)
            totalActivation = torch.sum(unweightedActivation, dim=1, keepdim=True)
            placeSignal = unweightedActivation / totalActivation
            return placeSignal.reshape([*agentLocation.shape[:2], -1])

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "hiddenObs": torch.zeros((self.linearHiddenSize,), dtype=torch.float32),
            "candidateGrid": torch.zeros((self.gridSize,), dtype=torch.float32),
            "hiddenGrid": torch.zeros((self.gridSize,), dtype=torch.float32),
        }

    def _getObsFromBatch(self, batch):
        obs = batch["obs"]
        visionSize = self.inputSize**2 * 3
        vision = obs[:, :, :visionSize]
        vision = torch.reshape(
            vision, [*vision.shape[:2], self.inputSize, self.inputSize, 3]
        ).to(torch.float32)
        agentLocation = obs[:, :, visionSize : visionSize + self.mazeSize**2]
        agentLocation = agentLocation.reshape(
            *agentLocation.shape[:2], self.mazeSize, self.mazeSize, 1
        ).to(torch.float32)
        action = obs[:, :, -5:].to(torch.float32)
        return vision, agentLocation, action

    def _processPreHeads(self, batch):
        vision, _, action = self._getObsFromBatch(batch)
        visionFeatures = self._processConvolution(vision)
        hiddenGrid = batch[Columns.STATE_IN]["hiddenGrid"]
        candidateGrid = batch[Columns.STATE_IN]["candidateGrid"]
        initialGridMask = (torch.sum(hiddenGrid, 1) == 0)[:, None]
        initialEncodedHidden: torch.Tensor = self.initialStates(
            torch.tensor([0], device=vision.device)
        )
        initialEncodedHidden = initialEncodedHidden.expand(
            vision.shape[0], initialEncodedHidden.shape[1]
        )
        initialEncodedCandidate: torch.Tensor = self.initialStates(
            torch.tensor([1], device=vision.device)
        )
        initialEncodedCandidate = initialEncodedCandidate.expand(
            vision.shape[0], initialEncodedCandidate.shape[1]
        )
        hiddenGrid = torch.where(initialGridMask, initialEncodedHidden, hiddenGrid)
        candidateGrid = torch.where(
            initialGridMask, initialEncodedCandidate, candidateGrid
        )
        gridStates, finalGridState = self.pathIntegrator(
            action, (hiddenGrid.unsqueeze(0), candidateGrid.unsqueeze(0))
        )
        initialHidden = batch[Columns.STATE_IN]["hiddenObs"].unsqueeze(0)
        gridStatesMatchingObsHidden = torch.concat(
            [hiddenGrid.unsqueeze(1), gridStates[:, :-1]], dim=1
        )
        decodedGrid = self.gridDecoder(gridStatesMatchingObsHidden)
        projectedPlace = self.placeProjector(decodedGrid)
        with torch.no_grad():
            decodedGridWithoutGrad = torch.Tensor(decodedGrid)
        visionAndGridFeatures = torch.concat(
            [visionFeatures, decodedGridWithoutGrad], dim=2
        )
        return (
            self.trajectoryMemory(visionAndGridFeatures, initialHidden)[0],
            projectedPlace,
            finalGridState,
        )

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        _, agentLocation, _ = self._getObsFromBatch(batch)
        hiddenStates, projectedPlace, finalGrid = self._processPreHeads(batch)
        policy = self.policy_branch(hiddenStates)
        return {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {
                "hiddenObs": hiddenStates[:, -1],
                "candidateGrid": finalGrid[1].squeeze(0),
                "hiddenGrid": finalGrid[0].squeeze(0),
            },
            Columns.EMBEDDINGS: hiddenStates,
            "placeLogit": projectedPlace,
            "placeTarget": self._calculatePlace(agentLocation),
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings, _, _ = self._processPreHeads(batch)
        return self.value_branch(embeddings).squeeze(-1)
