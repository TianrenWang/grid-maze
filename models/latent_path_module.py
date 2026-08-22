import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn

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
        self.manifoldProjector = nn.Sequential(
            nn.Linear(self.linearHiddenSize, 2), nn.Sigmoid()
        )
        self.manifoldPolicy = nn.Linear(2, self.action_space.n)
        self.manifoldValue = nn.Linear(2, 1)

    def _getPolicyAndValue(self, batch):
        vision, lastAgentLocation, _, _ = self._getObsFromBatch(batch)
        prevPlaces = self.placeEncoderForMemory(
            calculatePlace(self.placeCells, lastAgentLocation[:, 0, :])
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
        movements = None
        policy = self.policy_branch(memory)
        value = self.value_branch(memory)
        predictedPlaces = None
        actualPlaces = None
        finalGridState = None
        manifold = None

        def getOutputs():
            return (
                policy,
                value,
                predictedPlaces,
                actualPlaces,
                finalGridState,
                memory.detach(),
                movements,
                manifold,
            )

        selfLocalize = self.model_config.get("self_localize", False)
        learnManifold = self.model_config.get("learnManifold", False)

        if self.model_config.get("pretrain", False):
            return getOutputs()

        initialManifoldCoordinate = self.manifoldProjector(initialMemory)
        initialPlaceMask = torch.sum(batch[Columns.STATE_IN]["hiddenObs"], 1) == 0
        initialManifoldCoordinate = torch.where(
            initialPlaceMask[:, None],
            lastAgentLocation[:, 0, :],
            initialManifoldCoordinate,
        )
        manifoldCoordinates = self.manifoldProjector(memory)
        movements = torch.diff(
            torch.concat(
                [initialManifoldCoordinate.unsqueeze(1), manifoldCoordinates],
                dim=1,
            ),
            dim=1,
        )
        policy = self.manifoldPolicy(manifoldCoordinates)
        value = self.manifoldValue(manifoldCoordinates)
        manifold = manifoldCoordinates

        if learnManifold:
            return getOutputs()

        integratedCode, predictedPlaces, finalGridState = getIntegration(
            initialManifoldCoordinate, movements
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
            movements,
            manifold,
        ) = self._getPolicyAndValue(batch)
        output = {
            Columns.ACTION_DIST_INPUTS: policy,
            Columns.STATE_OUT: {
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

        if movements is not None:
            output["movements"] = movements

        if manifold is not None:
            output["manifold"] = manifold
            if "rewards" in batch:
                rewards = batch["rewards"]
                output["rewards"] = rewards

        return output

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            _, embeddings, _, _, _, _, _, _ = self._getPolicyAndValue(batch)
        return embeddings.squeeze(-1)
