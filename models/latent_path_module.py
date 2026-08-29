import math

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn

torch.set_printoptions(precision=2)

from .agent_models import PathIntegrationWithVisionModule
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


class LatentPathModule(PathIntegrationWithVisionModule):
    def setup(self):
        PathIntegrationWithVisionModule.setup(self)
        self.pathIntegrator = nn.LSTM(2, self.integratorSize, batch_first=True)
        self.directionDecoder = nn.Sequential(
            nn.Linear(self.linearHiddenSize + self.action_space.n + 1, 1), nn.Tanh()
        )
        self.speed = 1 / 30
        self.place_projector = nn.Sequential(
            nn.Linear(self.linearHiddenSize, self.numPlaceCells), nn.Softmax(dim=-1)
        )
        self.policy_branch = nn.Linear(self.numPlaceCells, self.action_space.n)
        self.value_branch = nn.Linear(self.numPlaceCells, 1)

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "hiddenObs": torch.zeros((self.linearHiddenSize,), dtype=torch.float32),
            "candidateGrid": torch.zeros((self.integratorSize,), dtype=torch.float32),
            "hiddenGrid": torch.zeros((self.integratorSize,), dtype=torch.float32),
            "manifoldCoordinate": torch.ones((2,), dtype=torch.float32),
        }

    def _getDisplacement(self, latent: torch.Tensor) -> torch.Tensor:
        angle = self.directionDecoder(latent) * math.pi
        theta = math.pi * angle
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1) * self.speed

    def _getPlaceActivationFromMemory(self, memory: torch.Tensor) -> torch.Tensor:
        placeActivation = self.place_projector(memory)
        return calculatePlace(
            self.placeCells,
            torch.matmul(placeActivation, self.placeCells),
            self.fieldSize,
        )

    def _getPolicyAndValue(self, batch):
        vision, lastAgentLocation, _, action = self._getObsFromBatch(batch)
        prevPlaces = self.placeEncoderForMemory(
            calculatePlace(self.placeCells, lastAgentLocation[:, 0, :], self.fieldSize)
        )
        initialMemory = self._getInitialMemory(
            prevPlaces, batch[Columns.STATE_IN]["hiddenObs"]
        )

        def getIntegration(startingCoordinate: torch.Tensor, movement: torch.Tensor):
            return self._pathIntegrate(
                startingCoordinate,
                movement,
                batch[Columns.STATE_IN]["hiddenGrid"],
                batch[Columns.STATE_IN]["candidateGrid"],
            )

        memory = self._processVisualMemory(vision, initialMemory)
        displacement = self._getDisplacement(torch.concat([memory, action], dim=-1))
        previousMemory = batch[Columns.STATE_IN]["hiddenObs"]
        initialCoordinateMask = torch.sum(previousMemory, 1) == 0
        startingPlace = torch.where(
            initialCoordinateMask[:, None],
            self.place_projector(prevPlaces) @ self.placeCells,
            batch[Columns.STATE_IN]["manifoldCoordinate"],
        )
        manifoldCoordinates = startingPlace.unsqueeze(1) + torch.cumsum(
            displacement, dim=1
        )
        placeActivation = calculatePlace(
            self.placeCells,
            manifoldCoordinates,
            self.fieldSize,
        )
        movements = None
        policy = self.policy_branch(placeActivation)
        value = self.value_branch(placeActivation)
        predictedPlaces = None
        actualPlaces = None
        finalGridState = None

        def getOutputs():
            return (
                policy,
                value,
                predictedPlaces,
                actualPlaces,
                finalGridState,
                memory.detach(),
                manifoldCoordinates,
            )

        selfLocalize = self.model_config.get("self_localize", False)
        learnManifold = self.model_config.get("learnManifold", False)

        if self.model_config.get("pretrain", False):
            return getOutputs()

        initialPlaceActivation = self._getPlaceActivationFromMemory(initialMemory)
        manifoldCoordinates = torch.matmul(
            torch.concat([initialPlaceActivation.unsqueeze(1), placeActivation], dim=1),
            self.placeCells,
        )
        movements = torch.diff(manifoldCoordinates, dim=1)

        if learnManifold:
            return getOutputs()

        integratedCode, predictedPlaces, finalGridState = getIntegration(
            initialPlaceActivation, movements
        )

        if selfLocalize:
            actualPlaces = calculatePlace(
                self.placeCells, self.EMAProjector(memory)[0]
            ).detach()
            return getOutputs()

        integration = self.gridCompressor(integratedCode)
        policy = self.piPolicyPredictor(integration)
        value = self.piValuePredictor(integration)

        return getOutputs()

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        (
            policy,
            value,
            predictedPlaces,
            actualPlaces,
            finalIntegrationState,
            memory,
            manifoldCoordinates,
        ) = self._getPolicyAndValue(batch)
        output = {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {
                "manifoldCoordinate": manifoldCoordinates[:, -1, :],
                "hiddenObs": memory[:, -1, :],
                "candidateGrid": finalIntegrationState[1].squeeze(0)
                if finalIntegrationState
                else batch[Columns.STATE_IN]["candidateGrid"],
                "hiddenGrid": finalIntegrationState[0].squeeze(0)
                if finalIntegrationState
                else batch[Columns.STATE_IN]["hiddenGrid"],
            },
            Columns.EMBEDDINGS: value,
        }
        if predictedPlaces is not None:
            output["placeLogit"] = predictedPlaces
            output["placeTarget"] = actualPlaces

        # if movements is not None:
        #     output["movements"] = movements

        return output

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            _, embeddings, _, _, _, _, _ = self._getPolicyAndValue(batch)
        return embeddings.squeeze(-1)
