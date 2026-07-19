import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.utils.annotations import override

from .agent_models import PlaceMazeModule
from .utils import calculatePlace


class ManifoldProjector(nn.Module):
    def __init__(self, inputSize: int, latentSize: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputSize, latentSize), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(latentSize, inputSize), nn.Tanh())

    def forward(self, latent: torch.Tensor):
        projection = self.encoder(latent)
        reconstructed = self.decoder(projection)
        return projection, reconstructed


class LatentPathModule(PlaceMazeModule):
    def setup(self):
        PlaceMazeModule.setup(self)
        self.pathIntegrator = nn.LSTM(2, self.integratorSize, batch_first=True)
        self.manifoldProjector = ManifoldProjector(self.linearHiddenSize, 2)

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _processPreHeads(self, batch):
        vision, lastAgentLocation, _, _ = self._getObsFromBatch(batch)
        initialMemory = self._getInitialMemory(
            lastAgentLocation[:, 0, :], batch[Columns.STATE_IN]["hiddenObs"]
        )

        def getIntegration(startingCoordinate: torch.Tensor, movement: torch.Tensor):
            return self._pathIntegrate(
                startingCoordinate,
                movement,
                batch[Columns.STATE_IN]["hiddenGrid"],
                batch[Columns.STATE_IN]["candidateGrid"],
            )

        memory = self._processVisualMemory(vision, initialMemory)

        predictedPlaces = None
        actualPlaces = None
        finalGridState = None
        reconstructedLatent = None
        movements = None

        if self.model_config.get("self_localize", False):
            memory = memory.detach()
            sequenceProjections, reconstructedLatent = self.manifoldProjector(
                torch.concat([initialMemory.unsqueeze(1), memory], dim=1)
            )
            movements = sequenceProjections[:, 1:, :] - sequenceProjections[:, :-1, :]
            gridCodes, predictedPlaces, finalGridState = getIntegration(
                sequenceProjections[:, 0, :], movements
            )
            accummulatedMovements = torch.cumsum(movements, dim=1)
            actualPositions = (
                sequenceProjections[:, 0, :][:, None, :] + accummulatedMovements
            )
            actualPlaces = calculatePlace(self.placeCells, actualPositions)
            """
            Doesn't make any sense to path integrate using artificially generated moves
            because the abstract tasks downstream won't have access to a convenient
            manifold to navigate smoothly in.
            """
            policyInput = memory
        elif self.model_config.get("pretraining", False):
            policyInput = memory
        else:
            gridCodes, _, _ = getIntegration()
            gate = self.gridGate(memory.detach())
            policyInput = memory * (1 - gate) + self.gridCompressor(gridCodes) * gate

        return (
            policyInput,
            predictedPlaces,
            actualPlaces.detach(),
            finalGridState,
            memory.detach(),
            reconstructedLatent[:, 1:, :],
            movements,
        )

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        (
            policyFeature,
            predictedPlaces,
            actualPlaces,
            finalGrid,
            memory,
            reconstructedLatent,
            movements,
        ) = self._processPreHeads(batch)
        policy = self.policy_branch(policyFeature)
        output = {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {
                "hiddenObs": memory[:, -1, :],
                "candidateGrid": finalGrid[1].squeeze(0)
                if finalGrid
                else batch[Columns.STATE_IN]["candidateGrid"],
                "hiddenGrid": finalGrid[0].squeeze(0)
                if finalGrid
                else batch[Columns.STATE_IN]["hiddenGrid"],
            },
            Columns.EMBEDDINGS: policyFeature,
        }
        if predictedPlaces is not None:
            output["placeLogit"] = predictedPlaces
            output["placeTarget"] = actualPlaces

        if reconstructedLatent is not None:
            output["reconstructedLatents"] = reconstructedLatent
            output["actualLatents"] = memory

        if movements is not None:
            output["movements"] = movements

        return output

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings, _, _, _, _, _, _ = self._processPreHeads(batch)
        return self.value_branch(embeddings).squeeze(-1)
