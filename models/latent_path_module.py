from dataclasses import dataclass

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn

from .agent_models import PathIntegrationWithVisionModule
from .jepa import JEPA
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


@dataclass
class ControlOutputs:
    policy: torch.Tensor
    value: torch.Tensor
    memory: torch.Tensor
    jepaMemory: torch.Tensor | None = None
    predictedPlaces: torch.Tensor | None = None
    actualPlaces: torch.Tensor | None = None
    finalIntegrationState: torch.Tensor | None = None
    jepaLoss: torch.Tensor | None = None


SELF_LOCALIZE = "self_localize"
LEARN_MANIFOLD = "learnManifold"
PRETRAIN = "pretrain"


class LatentPathModule(PathIntegrationWithVisionModule):
    def setup(self):
        PathIntegrationWithVisionModule.setup(self)
        self.pathIntegrator = nn.LSTM(2, self.integratorSize, batch_first=True)
        self.jepa = JEPA(self.inputSize, self.hiddenSize, self.action_space.n)
        self.placeEncoderForJEPA = nn.Linear(self.numPlaceCells, self.hiddenSize)

        if self.model_config.get("pretrain", False):
            self.trainingPhase = PRETRAIN
        elif self.model_config.get("learnManifold", False):
            self.trainingPhase = LEARN_MANIFOLD
        elif self.model_config.get("self_localize", False):
            self.trainingPhase = SELF_LOCALIZE
        else:
            self.trainingPhase = None

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "hiddenObs": torch.zeros((self.linearHiddenSize,), dtype=torch.float32),
            "candidateGrid": torch.zeros((self.integratorSize,), dtype=torch.float32),
            "hiddenGrid": torch.zeros((self.integratorSize,), dtype=torch.float32),
            "jepaMemory": torch.zeros((self.hiddenSize,), dtype=torch.float32),
        }

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
        policy = self.policy_branch(memory)
        value = self.value_branch(memory)

        output = ControlOutputs(policy=policy, value=value, memory=memory)

        if self.trainingPhase == PRETRAIN or "actions" not in batch:
            return output

        prevPlaces = self.placeEncoderForJEPA(
            calculatePlace(self.placeCells, lastAgentLocation[:, 0, :])
        )
        initialMemory = self._getInitialMemory(
            prevPlaces, batch[Columns.STATE_IN]["jepaMemory"]
        )
        jepaMemory, jepaLoss = self.jepa.forward_train(
            vision,
            initialMemory,
            torch.nn.functional.one_hot(
                batch["actions"].to(torch.long), num_classes=4
            ).to(torch.int32),
        )
        output.jepaMemory = jepaMemory
        output.jepaLoss = jepaLoss

        if self.trainingPhase == LEARN_MANIFOLD:
            return output

        return output

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        controlOutputs = self._getPolicyAndValue(batch)
        finalOutput = {
            Columns.ACTION_DIST_INPUTS: controlOutputs.policy,
            Columns.STATE_OUT: {"hiddenObs": controlOutputs.memory[:, -1, :]},
            Columns.EMBEDDINGS: controlOutputs.value,
        }

        stateOut = finalOutput[Columns.STATE_OUT]
        stateIn = batch[Columns.STATE_IN]

        if controlOutputs.finalIntegrationState is None:
            stateOut["candidateGrid"] = stateIn["candidateGrid"]
            stateOut["hiddenGrid"] = stateIn["hiddenGrid"]
        else:
            stateOut["candidateGrid"] = controlOutputs.finalIntegrationState[1].squeeze(
                0
            )
            stateOut["hiddenGrid"] = controlOutputs.finalIntegrationState[0].squeeze(0)

        if controlOutputs.jepaMemory is None:
            stateOut["jepaMemory"] = stateIn["jepaMemory"]
        else:
            stateOut["jepaMemory"] = controlOutputs.jepaMemory[:, -1, :]

        if type(controlOutputs.predictedPlaces) is torch.Tensor:
            finalOutput["placeLogit"] = controlOutputs.predictedPlaces
            finalOutput["placeTarget"] = controlOutputs.actualPlaces

        if type(controlOutputs.jepaLoss) is torch.Tensor:
            finalOutput["jepaLoss"] = controlOutputs.jepaLoss

        return finalOutput

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self._getPolicyAndValue(batch).value
        return embeddings.squeeze(-1)
