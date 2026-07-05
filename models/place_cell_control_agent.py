import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns

from .agent_models import MemoryMazeModule
from .utils import calculatePlace


class PlaceCellControlModule(MemoryMazeModule):
    def setup(self):
        MemoryMazeModule.setup(self)
        self.numPlaceCells = self.model_config.get("numPlaceCells", 32)
        self.placeEncoder = nn.Linear(self.numPlaceCells, self.linearHiddenSize)
        self.placeCells = nn.Parameter(torch.rand([self.numPlaceCells, 2]), False)

    def _getObsFromBatch(self, batch):
        obs = batch["obs"]
        visionSize = self.inputSize**2 * 2
        vision = obs[:, :, :visionSize]
        vision = torch.reshape(
            vision, [*vision.shape[:2], self.inputSize, self.inputSize, 2]
        )
        lastAgentLocation = obs[:, :, visionSize : visionSize + 2]
        lastAgentLocation = lastAgentLocation.reshape(*lastAgentLocation.shape[:2], 2)
        return vision, lastAgentLocation

    def _processPreHeads(self, batch):
        initialHidden: torch.Tensor = batch[Columns.STATE_IN]["hiddenObs"]
        vision, lastAgentLocation = self._getObsFromBatch(batch)
        prevPlaces = self.placeEncoder(
            calculatePlace(self.placeCells, lastAgentLocation)[:, 0, :]
        )
        initialPlaceMask = torch.sum(initialHidden, 1) == 0
        initialHidden = torch.where(
            initialPlaceMask[:, None], prevPlaces, initialHidden
        )
        visionFeatures = self._processConvolution(vision)
        return self.trajectoryMemory(visionFeatures, initialHidden.unsqueeze(0))
