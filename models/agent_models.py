import torch
import torch.nn as nn
import math
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
        self.policy_branch = nn.Linear(
            self.linearHiddenSize, self.action_space.shape[0]
        )
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
        self.memoryEncoder = nn.Sequential(
            nn.Linear(self.linearHiddenSize + self.gridSize, self.linearHiddenSize),
            nn.ReLU(),
        )
        self.initialStates = nn.Embedding(2, self.integratorSize)
        self.placeCells = nn.Parameter(torch.rand([self.numPlaceCells, 2]), False)
        self.fieldSize = self.mazeSize / math.sqrt(self.numPlaceCells) * 0.01
        self.placeEncoder = nn.Linear(self.numPlaceCells, 2 * self.integratorSize)

    def _calculatePlace(self, agentLocation: torch.Tensor):
        with torch.no_grad():
            agentLocationShape = agentLocation.shape
            agentLocation = agentLocation.flatten(0, -2)
            diff = agentLocation.unsqueeze(1) - self.placeCells.unsqueeze(0)
            dists_squared = torch.sum(torch.abs(diff) ** 2, dim=-1)
            unnormalized_activations = -dists_squared / (2 * self.fieldSize**2)
            normalized_activations = torch.nn.functional.softmax(
                unnormalized_activations, dim=1
            )
            return normalized_activations.reshape(
                [*agentLocationShape[:2], self.numPlaceCells]
            )

    """
    Following code is used to figure out how dramatically place code can change
    between moving squares.

    def _calculatePlace(self, agentLocation: torch.Tensor):
        coordinates = [[[i, j] for i in range(31)] for j in range(31)]
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        coordinates = coordinates / 30
        coordinates = coordinates.flatten(0, -2)
        with torch.no_grad():
            agentLocationShape = agentLocation.shape
            agentLocation = coordinates
            diff = agentLocation.unsqueeze(1) - self.placeCells.unsqueeze(0)
            dists_squared = torch.sum(torch.abs(diff) ** 2, dim=-1)
            unnormalized_activations = -dists_squared / (2 * self.fieldSize**2)
            normalized_activations = torch.nn.functional.softmax(
                unnormalized_activations, dim=1
            )
            normDiff = normalized_activations.unsqueeze(
                1
            ) - normalized_activations.unsqueeze(0)
            normDiffSum = torch.sum(torch.abs(normDiff), dim=2)
            sorted, _ = torch.sort(normDiffSum)
            print(torch.round(torch.mean(sorted, dim=0), decimals=2)[:20])
            raise Exception("dogshit")
            return normalized_activations.reshape(
                [*agentLocationShape[:2], self.numPlaceCells]
            )
    """

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "hiddenObs": torch.zeros((self.linearHiddenSize,), dtype=torch.float32),
            "candidateGrid": torch.zeros((self.integratorSize,), dtype=torch.float32),
            "hiddenGrid": torch.zeros((self.integratorSize,), dtype=torch.float32),
        }

    def _getObsFromBatch(self, batch):
        obs = batch["obs"]
        visionSize = self.inputSize**2 * 3
        vision = obs[:, :, :visionSize]
        vision = torch.reshape(
            vision, [*vision.shape[:2], self.inputSize, self.inputSize, 3]
        )
        lastAgentLocation = obs[:, :, visionSize : visionSize + 2]
        lastAgentLocation = lastAgentLocation.reshape(*lastAgentLocation.shape[:2], 2)
        agentLocation = obs[:, :, visionSize + 2 : visionSize + 4]
        agentLocation = agentLocation.reshape(*agentLocation.shape[:2], 2)
        action = obs[:, :, -5:]
        return vision, lastAgentLocation, agentLocation, action

    def _processPreHeads(self, batch, eval: bool = False):
        vision, lastAgentLocation, _, action = self._getObsFromBatch(batch)
        visionFeatures = self._processConvolution(vision)
        prevPlaces = self.placeEncoder(self._calculatePlace(lastAgentLocation)[:, 0, :])
        hiddenGrid = prevPlaces[:, : self.integratorSize].contiguous()
        candidateGrid = prevPlaces[:, self.integratorSize :].contiguous()
        if eval:
            hiddenPlace = hiddenGrid
            candidatePlace = candidateGrid
            hiddenGrid = batch[Columns.STATE_IN]["hiddenGrid"]
            candidateGrid = batch[Columns.STATE_IN]["candidateGrid"]
            initialPlaceMask = torch.sum(hiddenGrid, 1) == 0
            randomPlaceMask = (
                torch.rand(initialPlaceMask.shape, dtype=torch.float32) < 0.05
            )
            placeMask = torch.where(randomPlaceMask, randomPlaceMask, initialPlaceMask)[
                :, None
            ]
            hiddenGrid = torch.where(placeMask, hiddenPlace, hiddenGrid)
            candidateGrid = torch.where(placeMask, candidatePlace, candidateGrid)
        gridStates, finalGridState = self.pathIntegrator(
            action, (hiddenGrid.unsqueeze(0), candidateGrid.unsqueeze(0))
        )
        initialHidden = batch[Columns.STATE_IN]["hiddenObs"].unsqueeze(0)
        decodedGrid = self.gridDecoder(gridStates)
        projectedPlace = self.placeProjector(decodedGrid)
        with torch.no_grad():
            decodedGridWithoutGrad = torch.Tensor(decodedGrid)
        visionAndGridFeatures = torch.concat(
            [visionFeatures, decodedGridWithoutGrad], dim=2
        )
        visionAndGridFeatures = self.memoryEncoder(visionAndGridFeatures)
        return (
            self.trajectoryMemory(visionAndGridFeatures, initialHidden)[0],
            projectedPlace,
            finalGridState,
        )

    def _forward_exploration(self, batch, **kwargs):
        hiddenStates, _, finalGrid = self._processPreHeads(batch, True)
        policy = self.policy_branch(hiddenStates)
        return {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {
                "hiddenObs": hiddenStates[:, -1],
                "candidateGrid": finalGrid[1].squeeze(0),
                "hiddenGrid": finalGrid[0].squeeze(0),
            },
            Columns.EMBEDDINGS: hiddenStates,
        }

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        _, _, agentLocation, _ = self._getObsFromBatch(batch)
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
            "placeCells": self.placeCells.unsqueeze(0)
            .unsqueeze(0)
            .expand([*projectedPlace.shape[:2], self.numPlaceCells, 2]),
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings, _, _ = self._processPreHeads(batch)
        return self.value_branch(embeddings).squeeze(-1)


class GPSModule(MemoryMazeModule):
    def setup(self):
        MemoryMazeModule.setup(self)
        self.trajectoryMemory = nn.GRU(
            self.linearHiddenSize + 2, self.linearHiddenSize, batch_first=True
        )

    def _getObsFromBatch(self, batch):
        obs = batch["obs"]
        visionSize = self.inputSize**2 * 3
        vision = obs[:, :, :visionSize]
        vision = torch.reshape(
            vision, [*vision.shape[:2], self.inputSize, self.inputSize, 3]
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


class VectorPredictor(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.speedPredictor = nn.Sequential(nn.Linear(num_hidden, 1), nn.Sigmoid())
        self.rotationPredictor = nn.Sequential(nn.Linear(num_hidden, 1), nn.Tanh())
        self.stdLogPredictor = nn.Sequential(nn.Linear(num_hidden, 2))

    def forward(self, logit):
        speed = self.speedPredictor(logit)
        rotation = self.rotationPredictor(logit) * 3.14
        stglog = self.stdLogPredictor(logit)
        return torch.concat([speed, rotation, stglog], dim=2)


class ContinuousMazeModule(PlaceMazeModule):
    def setup(self):
        PlaceMazeModule.setup(self)
        self.policy_branch = VectorPredictor(self.linearHiddenSize)
        self.pathIntegrator = nn.LSTM(3, self.integratorSize, batch_first=True)

    def _getObsFromBatch(self, batch):
        obs = batch["obs"]
        visionSize = self.inputSize**2 * 3
        vision = obs[:, :, :visionSize]
        vision = torch.reshape(
            vision, [*vision.shape[:2], self.inputSize, self.inputSize, 3]
        )
        lastAgentLocation = obs[:, :, visionSize : visionSize + 2]
        lastAgentLocation = lastAgentLocation.reshape(*lastAgentLocation.shape[:2], 2)
        agentLocation = obs[:, :, visionSize + 2 : visionSize + 4]
        agentLocation = agentLocation.reshape(*agentLocation.shape[:2], 2)
        action = obs[:, :, -3:]
        return vision, lastAgentLocation, agentLocation, action
