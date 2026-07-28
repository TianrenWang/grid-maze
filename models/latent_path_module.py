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
        self.manifoldProjector = ManifoldProjector(self.linearHiddenSize, 2)

    def _getPolicyAndValue(self, batch):
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

        policy = self.policy_branch(memory)
        value = self.value_branch(memory)
        predictedPlaces = None
        actualPlaces = None
        finalGridState = None
        reconstructedLatent = None
        movements = None

        def getOutputs():
            return (
                policy,
                value,
                predictedPlaces,
                actualPlaces,
                finalGridState,
                memory.detach(),
                reconstructedLatent,
                movements,
            )

        selfLocalize = self.model_config.get("self_localize", False)
        useIntegrationPolicy = self.model_config.get("integrationPolicy", False)
        learnProjector = self.model_config.get("learn_projector", False)

        if selfLocalize or useIntegrationPolicy:
            memory = memory.detach()
            initialMemory = initialMemory.detach()
            sequenceProjections, reconstructedLatent = self.manifoldProjector(
                torch.concat([initialMemory.unsqueeze(1), memory], dim=1)
            )
            reconstructedLatent = reconstructedLatent[:, 1:, :]
            if learnProjector:
                return getOutputs()
            movements = sequenceProjections[:, 1:, :] - sequenceProjections[:, :-1, :]
            integratedCode, predictedPlaces, finalGridState = getIntegration(
                sequenceProjections[:, 0, :], movements
            )
            actualPlaces = calculatePlace(
                self.placeCells, sequenceProjections[:, 1:, :]
            ).detach()
            if useIntegrationPolicy:
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
            reconstructedLatent,
            movements,
        ) = self._processPreHeads(batch)
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

        if reconstructedLatent is not None:
            output["reconstructedLatents"] = reconstructedLatent
            output["actualLatents"] = memory

        if movements is not None:
            output["movements"] = movements

        return output

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            _, embeddings, _, _, _, _, _, _ = self._processPreHeads(batch)
        return embeddings.squeeze(-1)
