import torch
import torch.nn as nn
import math
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.utils.annotations import override

from .agent_models import MemoryMazeModule
from .multi_head_lstm import MultiHeadLSTM

NUM_MODULES = 2
GRID_MODULE_DIM = 2


class ModuleProjector(nn.Module):
    def __init__(self, latentSize: int, alpha: float):
        super().__init__()
        self.count = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.mean = nn.Parameter(
            torch.zeros(latentSize, dtype=torch.float32), requires_grad=False
        )
        self.covariance = nn.Parameter(
            torch.zeros([latentSize, latentSize], dtype=torch.float32),
            requires_grad=False,
        )
        self.principalComponents = nn.Parameter(
            torch.zeros(
                [latentSize, NUM_MODULES * GRID_MODULE_DIM], dtype=torch.float32
            ),
            requires_grad=False,
        )

    def update(self, latent: torch.Tensor):
        with torch.no_grad():
            currentMean = latent.mean(dim=0)
            self.mean.copy_(currentMean * self.alpha + self.mean * (1 - self.alpha))
            centered = latent - self.mean[None, :]
            currentCovariance = centered.T @ centered / (latent.shape[0] - 1)
            self.covariance.copy_(
                currentCovariance * self.alpha + self.covariance * (1 - self.alpha)
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(self.covariance)
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, idx]
            self.principalComponents.copy_(
                eigenvectors[:, : NUM_MODULES * GRID_MODULE_DIM]
            )

    def forward(self, latent: torch.Tensor):
        return (latent - self.mean) @ self.principalComponents


class LatentPathModule(MemoryMazeModule):
    def setup(self):
        MemoryMazeModule.setup(self)
        self.integratorSize = 128
        self.gridSize = 512
        self.numPlaceCells = 32
        self.mazeSize = self.model_config.get("mazeSize", 31)
        self.gridCompressor = nn.Sequential(
            nn.Linear(self.gridSize * NUM_MODULES, self.gridSize),
            nn.ReLU(),
            nn.Linear(self.gridSize, self.linearHiddenSize),
            nn.ReLU(),
        )
        self.memoryEncoder = nn.Sequential(
            nn.Linear(
                self.linearHiddenSize * 2,
                self.linearHiddenSize,
            ),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.moduleProjector = ModuleProjector(self.linearHiddenSize, 0.01)
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
            (torch.rand([NUM_MODULES, self.numPlaceCells, GRID_MODULE_DIM]) - 0.5) * 2,
            False,
        )
        self.fieldSize = 0.3 / math.sqrt(self.numPlaceCells)
        self.placeEncoderWeight = nn.Parameter(
            torch.Tensor(NUM_MODULES, self.numPlaceCells, 2 * self.integratorSize)
        )
        self.placeEncoderBias = nn.Parameter(
            torch.Tensor(NUM_MODULES, 2 * self.integratorSize)
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

    def get_place_activation(self, coordinates: torch.Tensor):
        with torch.no_grad():
            originalShape = coordinates.shape[:-2]
            diff = coordinates.flatten(0, -3).unsqueeze(2) - self.placeCells.unsqueeze(
                0
            )
            dist2 = (diff**2).sum(dim=-1)
            return torch.nn.functional.softmax(
                -dist2 / (2 * self.fieldSize**2), dim=-1
            ).reshape([*originalShape, NUM_MODULES, self.numPlaceCells])

    def _pathIntegrate(
        self,
        latents: torch.Tensor,
        initialLatents: torch.Tensor,
    ):
        currentProjections = self.moduleProjector.forward(latents).reshape(
            [*latents.shape[:2], NUM_MODULES, GRID_MODULE_DIM]
        )
        pastProjections = self.moduleProjector.forward(initialLatents).reshape(
            [initialLatents.shape[0], NUM_MODULES, GRID_MODULE_DIM]
        )
        pastPlaceCellActivations = self.get_place_activation(pastProjections)
        encodedPlace = torch.einsum(
            "bmp,mpe->bme", pastPlaceCellActivations, self.placeEncoderWeight
        ) + self.placeEncoderBias.unsqueeze(0)
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
            "btmg,mgp->btmp",
            torch.nn.functional.dropout(gridCodes),
            self.placeDecoderWeight,
        )
        return (
            gridCodes.flatten(2),
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

    def _getPolicyInput(self, features, gridCode):
        with torch.no_grad():
            gridWithoutGrad = torch.Tensor(gridCode)
        compressedGrid = self.gridCompressor.forward(gridWithoutGrad)
        encodedMemory = self.memoryEncoder(
            torch.concat([features, compressedGrid], dim=2)
        )
        return encodedMemory

    def _processPreHeads(self, batch):
        initialLatent = batch[Columns.STATE_IN]["hiddenObs"]
        latents, _ = super()._processPreHeads(batch)
        with torch.no_grad():
            latentsWithoutGrad = torch.Tensor(latents)
        if self.model_config.get("self_localize"):
            self.moduleProjector.update(
                torch.concat([latents[batch["loss_mask"]], initialLatent], dim=0)
            )
        gridCode, predictedPlaces, actualPlaces, finalGrid = self._pathIntegrate(
            latentsWithoutGrad, initialLatent
        )
        return (
            self._getPolicyInput(latents, torch.nn.functional.dropout(gridCode)),
            predictedPlaces,
            actualPlaces,
            finalGrid,
            latentsWithoutGrad,
        )

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        (
            policyFeature,
            predictedPlaces,
            actualPlaces,
            finalGrid,
            latents,
        ) = self._processPreHeads(batch)
        policy = self.policy_branch(policyFeature)
        return {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {
                "hiddenObs": latents[:, -1, :],
                "candidateGrid": finalGrid[1].squeeze(0),
                "hiddenGrid": finalGrid[0].squeeze(0),
            },
            Columns.EMBEDDINGS: policyFeature,
            "placeLogit": predictedPlaces,
            "placeTarget": actualPlaces,
            "latents": latents,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings, _, _, _, _ = self._processPreHeads(batch)
        return self.value_branch(embeddings).squeeze(-1)
