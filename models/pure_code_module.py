import torch.nn as nn
from ray.rllib.core.columns import Columns

from .agent_models import PlaceMazeModule


class PureCodeModule(PlaceMazeModule):
    def setup(self):
        PlaceMazeModule.setup(self)
        self.prePolicyEncoder = nn.Sequential(
            nn.Linear(self.gridSize, self.linearHiddenSize),
            nn.ReLU(),
        )
        self.placeEncoder = nn.Linear(self.numPlaceCells, 2 * self.integratorSize)

    def _processPreHeads(self, batch):
        _, lastAgentLocation, _, action = self._getObsFromBatch(batch)
        gridCodes, projectedPlace, finalGridState = self._pathIntegrate(
            lastAgentLocation,
            action,
            batch[Columns.STATE_IN]["hiddenGrid"],
            batch[Columns.STATE_IN]["candidateGrid"],
        )
        policyInput = self.prePolicyEncoder(gridCodes)
        return policyInput, policyInput, projectedPlace, finalGridState
