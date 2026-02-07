import torch
import torch.nn as nn
import math
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.utils.annotations import override

from .agent_models import MemoryMazeModule
from .simple_conv import SimpleConv
from .multi_head_lstm import MultiHeadLSTM

NUM_MODULES = 2
GRID_MODULE_DIM = 2


class LatentPathModule(MemoryMazeModule):
    def setup(self):
        MemoryMazeModule.setup(self)
        self.integratorSize = 128
        self.gridSize = 512
        self.numPlaceCells = 32
        self.mazeSize = self.model_config.get("mazeSize", 31)
        self.memoryEncoder = nn.Sequential(
            nn.Linear(
                self.linearHiddenSize + self.gridSize * NUM_MODULES,
                self.linearHiddenSize,
            ),
            nn.ReLU(),
        )

        self.gridConvModule = SimpleConv(self.hiddenSize)
        self.preGridHead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.primaryConvModuleOutSize**2 * self.hiddenSize * 2,
                self.linearHiddenSize,
            ),
            nn.ReLU(),
        )

        self.moduleProjector = nn.Linear(
            self.linearHiddenSize, GRID_MODULE_DIM * NUM_MODULES, bias=False
        )
        self.pathIntegrator = MultiHeadLSTM(
            GRID_MODULE_DIM, self.integratorSize, NUM_MODULES
        )
        self.gridProjectorWeight = nn.Parameter(
            torch.Tensor(NUM_MODULES, self.integratorSize, self.gridSize)
        )
        self.gridProjectorBias = nn.Parameter(torch.Tensor(NUM_MODULES, self.gridSize))
        self.placeDecoderWeight = nn.Parameter(
            torch.Tensor(NUM_MODULES, self.gridSize, self.numPlaceCells)
        )
        self.placeCells = nn.Parameter(
            torch.rand([NUM_MODULES, self.numPlaceCells, GRID_MODULE_DIM]), False
        )
        self.fieldSize = 0.3 / math.sqrt(self.numPlaceCells)
        self.placeEncoderWeight = nn.Parameter(
            torch.Tensor(NUM_MODULES, self.numPlaceCells, 2 * self.integratorSize)
        )
        self.placeEncoderBias = nn.Parameter(
            torch.Tensor(NUM_MODULES, self.numPlaceCells)
        )
        self.actionPredictor = nn.Sequential(
            nn.Linear(self.gridSize * NUM_MODULES, self.gridSize),
            nn.ReLU(),
            nn.Linear(self.gridSize, 5),
        )
        stdv = 1.0 / math.sqrt(self.integratorSize)
        for w in (
            self.gridProjectorWeight,
            self.gridProjectorBias,
            self.placeDecoderWeight,
            self.placeEncoderWeight,
            self.placeEncoderBias,
        ):
            nn.init.uniform_(w, -stdv, stdv)

    def _processConvolutionForGrid(self, vision: torch.Tensor):
        vision = vision.permute(0, 3, 1, 2).to(torch.float32)
        visionFeatures = self.gridConvModule(vision)
        visionFeatures = self.preGridHead(visionFeatures)
        return visionFeatures

    def get_place_activation(self, coordinates: torch.Tensor):
        originalShape = coordinates.shape[:-2]
        diff = coordinates.flatten(0, -3).unsqueeze(2) - self.placeCells.unsqueeze(0)
        dist2 = (diff**2).sum(dim=-1)
        return torch.nn.functional.softmax(
            torch.exp(-dist2 / (2 * self.fieldSize**2)), dim=-1
        ).reshape([*originalShape, NUM_MODULES, self.numPlaceCells])

    def _pathIntegrate(self, vision: torch.Tensor, previousVision: torch.Tensor):
        batchSize = vision.shape[0]
        sequenceLength = vision.shape[1]
        previousVision = previousVision[:, 0, :]
        latents = self._processConvolutionForGrid(
            torch.concat([vision.flatten(0, 1), previousVision])
        )
        modularProjections = self.moduleProjector.forward(latents) % 1
        currentProjections = (
            modularProjections[:-batchSize]
            .contiguous()
            .reshape([batchSize, sequenceLength, NUM_MODULES, GRID_MODULE_DIM])
        )
        pastProjections = (
            modularProjections[-batchSize:]
            .contiguous()
            .reshape([batchSize, NUM_MODULES, GRID_MODULE_DIM])
        ) % 1
        pastPlaceCellActivations = self.get_place_activation(pastProjections)
        encodedPlace = torch.einsum(
            "bmp,mpe->bme", pastPlaceCellActivations, self.placeEncoderWeight
        )
        initialHidden = encodedPlace[:, :, : self.integratorSize].contiguous()
        initialCandidate = encodedPlace[:, :, self.integratorSize :].contiguous()
        movements = currentProjections - torch.concat(
            [
                pastProjections[:, None, :, :],
                currentProjections[:, :-1, :, :],
            ],
            dim=1,
        )
        hiddens, finalStates = self.pathIntegrator.forward(
            movements, initialHidden, initialCandidate, 20
        )
        gridCodes = (
            torch.einsum("btmd,mdg->btmg", hiddens, self.gridProjectorWeight)
            + self.gridProjectorBias[None, None, :, :]
        )
        predictedPlaceLogit = torch.einsum(
            "btmg,mgp->btmp", gridCodes, self.placeDecoderWeight
        )
        return (
            gridCodes.flatten(2, 3),
            predictedPlaceLogit,
            self.get_place_activation(currentProjections),
            finalStates,
        )

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "hiddenObs": torch.zeros((self.linearHiddenSize,), dtype=torch.float32),
            "candidateGrid": torch.zeros(
                (NUM_MODULES * self.integratorSize,), dtype=torch.float32
            ),
            "hiddenGrid": torch.zeros(
                (NUM_MODULES * self.integratorSize,), dtype=torch.float32
            ),
        }

    def _getObsFromBatch(self, batch):
        obs = batch["obs"]
        visionSize = self.inputSize**2 * 3
        vision = obs[:, :, :visionSize]
        previousVision = obs[:, :, visionSize:-5]
        visionShape = [*vision.shape[:2], self.inputSize, self.inputSize, 3]
        vision = torch.reshape(vision, visionShape)
        previousVision = torch.reshape(previousVision, visionShape)
        return vision, previousVision, obs[:, :, -5:]

    # def _forward_exploration(self, batch, **kwargs):
    #     hiddenStates, _, finalGrid = self._processPreHeads(batch, True)
    #     policy = self.policy_branch(hiddenStates)
    #     return {
    #         Columns.ACTION_DIST_INPUTS: policy,
    #         Columns.STATE_OUT: {
    #             "hiddenObs": hiddenStates[:, -1],
    #             "candidateGrid": finalGrid[1].squeeze(0),
    #             "hiddenGrid": finalGrid[0].squeeze(0),
    #         },
    #         Columns.EMBEDDINGS: hiddenStates,
    #     }

    def _getPolicyInput(self, features, gridCode, initialHidden):
        with torch.no_grad():
            gridWithoutGrad = torch.Tensor(gridCode)
        encodedMemory = self.memoryEncoder(
            torch.concat([features, gridWithoutGrad], dim=2)
        )
        encodedMemory = self.trajectoryMemory(encodedMemory, initialHidden)
        return encodedMemory[0]

    def _processPreHeads(self, batch):
        vision, previousVision, action = self._getObsFromBatch(batch)
        visionFeatures = self._processConvolution(vision)
        gridCode, predictedPlaces, actualPlaces, finalGrid = self._pathIntegrate(
            vision, previousVision
        )
        initialHidden = batch[Columns.STATE_IN]["hiddenObs"].unsqueeze(0)
        hiddenStates = self._getPolicyInput(visionFeatures, gridCode, initialHidden)
        predictedAction = self.actionPredictor.forward(gridCode)
        return (
            hiddenStates,
            predictedPlaces,
            actualPlaces,
            finalGrid,
            predictedAction,
        )

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        hiddenStates, predictedPlaces, actualPlaces, finalGrid, predictedAction = (
            self._processPreHeads(batch)
        )
        policy = self.policy_branch(hiddenStates)
        return {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {
                "hiddenObs": hiddenStates[:, -1],
                "candidateGrid": finalGrid[1].squeeze(0),
                "hiddenGrid": finalGrid[0].squeeze(0),
            },
            Columns.EMBEDDINGS: hiddenStates,
            "placeLogit": predictedPlaces,
            "placeTarget": actualPlaces,
            "predictedAction": predictedAction,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings, _, _, _, _ = self._processPreHeads(batch)
        return self.value_branch(embeddings).squeeze(-1)
