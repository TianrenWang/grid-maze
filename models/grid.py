import torch.nn.functional as F
import torch.nn as nn


class FCBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.fc = nn.Linear(num_hidden, num_hidden)
        self.bn = nn.BatchNorm1d(num_hidden)

    def forward(self, x):
        return F.relu(self.bn(self.fc(x)))


class PlaceProcessor(nn.Module):
    def __init__(self, cells, hiddenSize, layers):
        super().__init__()
        self.cells = cells
        self.hiddenSize = hiddenSize
        self.startBlock = nn.Sequential(
            nn.Linear(cells, hiddenSize), nn.BatchNorm1d(hiddenSize), nn.ReLU()
        )
        self.processor = nn.ModuleList([FCBlock(hiddenSize) for _ in range(layers)])

    def forward(self, x):
        x = self.startBlock(x)
        for module in self.processor:
            x = module(x)
        return x
