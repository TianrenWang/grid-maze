from dataclasses import dataclass

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn

torch.set_printoptions(precision=2)

from .agent_models import PathIntegrationWithVisionModule
from .constants import *
from .jepa import JEPA
from .manifold_projector import ManifoldProjector
from .utils import calculatePlace


@dataclass
class ControlOutputs:
    policy: torch.Tensor | None = None
    value: torch.Tensor | None = None
    memory: torch.Tensor | None = None
    jepaMemory: torch.Tensor | None = None
    predictedPlaces: torch.Tensor | None = None
    actualPlaces: torch.Tensor | None = None
    finalIntegrationState: torch.Tensor | None = None
    jepaLoss: torch.Tensor | None = None
    coordinateReadout: torch.Tensor | None = None
    projectedManifold: torch.Tensor | None = None
    calculatedManifold: torch.Tensor | None = None
    jepaLatent: torch.Tensor | None = None


class LatentPathModule(PathIntegrationWithVisionModule):
    def setup(self):
        PathIntegrationWithVisionModule.setup(self)
        self.pathIntegrator = nn.LSTM(
            MANIFOLD_DIM, self.integratorSize, batch_first=True
        )
        self.jepa = JEPA(self.inputSize, self.hiddenSize, int(self.action_space.n) + 1)
        self.manifoldProjector = ManifoldProjector(
            self.hiddenSize, int(self.action_space.n) + 1, MANIFOLD_DIM
        )

        if self.model_config.get(LEARN_JEPA, False):
            self.trainingPhase = LEARN_JEPA
        elif self.model_config.get(VISION_POLICY, False):
            self.trainingPhase = VISION_POLICY
        elif self.model_config.get(LEARN_MANIFOLD, False):
            self.trainingPhase = LEARN_MANIFOLD
        elif self.model_config.get(SELF_LOCALIZE, False):
            self.trainingPhase = SELF_LOCALIZE
        elif self.model_config.get(LEARN_MOVEMENT, False):
            self.trainingPhase = LEARN_MOVEMENT
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

    def _getPlaceActivationFromMemory(self, memory: torch.Tensor) -> torch.Tensor:
        placeActivation = self.place_projector(memory)
        return calculatePlace(
            self.placeCells,
            torch.matmul(placeActivation, self.placeCells),
            self.fieldSize,
        )

    def _getPolicyAndValue(self, batch):
        vision, _, _, action = self._getObsFromBatch(batch)
        output = ControlOutputs()

        if self.trainingPhase not in POLICY_LEARNING:
            if self.trainingPhase == LEARN_JEPA:
                jepaMemory, jepaLoss, coordinateReadout = self.jepa.forward_train(
                    vision,
                    action,
                )
                output.coordinateReadout = coordinateReadout
                output.jepaMemory = jepaMemory
                output.jepaLoss = jepaLoss
            elif (
                self.trainingPhase == LEARN_MOVEMENT
                or self.trainingPhase == LEARN_MANIFOLD
            ):
                jepaLatent = self.jepa.forward(vision)
                projectedManifold, calculatedManifold = (
                    self.manifoldProjector.forward_train(jepaLatent, action)
                )
                output.jepaLatent = jepaLatent
                output.projectedManifold = projectedManifold
                output.calculatedManifold = calculatedManifold

            obs = batch["obs"]
            output.policy = torch.ones(
                [*obs.shape[:2], self.action_space.n],
                dtype=torch.float32,
                device=obs.device,
            )
            output.memory = torch.randn(
                [*obs.shape[:2], self.linearHiddenSize],
                dtype=torch.float32,
                device=obs.device,
            )
            output.value = torch.ones(
                [*obs.shape[:2], 1], dtype=torch.float32, device=obs.device
            )

            return output
        elif self.trainingPhase == VISION_POLICY:
            memory = self._processVisualMemory(
                vision, batch[Columns.STATE_IN]["hiddenObs"]
            )
            output.policy = self.policy_branch(memory)
            output.value = self.value_branch(memory)
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

        if controlOutputs.coordinateReadout is not None:
            finalOutput["coordinateReadout"] = controlOutputs.coordinateReadout

        if controlOutputs.predictedPlaces:
            finalOutput["placeLogit"] = controlOutputs.predictedPlaces
            finalOutput["placeTarget"] = controlOutputs.actualPlaces

        if controlOutputs.jepaLoss is not None:
            finalOutput["jepaLoss"] = controlOutputs.jepaLoss

        if controlOutputs.projectedManifold is not None:
            finalOutput["projectedManifold"] = controlOutputs.projectedManifold
            finalOutput["calculatedManifold"] = controlOutputs.calculatedManifold

        if controlOutputs.jepaLatent is not None:
            finalOutput["jepaLatent"] = controlOutputs.jepaLatent

        return finalOutput

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self._getPolicyAndValue(batch).value
        return embeddings.squeeze(-1)
