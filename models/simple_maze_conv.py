import torch.nn as nn


class SimpleMazeConv(nn.Module):
    def __init__(self, hiddenSize: int):
        super().__init__()
        self.convModule = nn.Sequential(
            nn.Conv2d(3, hiddenSize, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hiddenSize, hiddenSize * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.convModule(x)
