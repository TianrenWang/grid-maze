import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self, hiddenSize: int, inputSize: int = 3):
        super().__init__()
        self.convModule = nn.Sequential(
            nn.Conv2d(inputSize, hiddenSize, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hiddenSize, hiddenSize * 2, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                hiddenSize * 2, hiddenSize * 2, kernel_size=3, padding=1, stride=2
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.convModule(x)
