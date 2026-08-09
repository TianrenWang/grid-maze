import torch
from torch import nn


class SimpleConv(nn.Module):
    def __init__(self, hiddenSize: int):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(2, hiddenSize, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hiddenSize, hiddenSize * 2, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                hiddenSize * 2, hiddenSize * 2, kernel_size=3, padding=1, stride=2
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape

        assert len(shape) > 4, "SimpleConv was expecting a time dimension"

        x = x.flatten(0, -4)
        x = x.permute(0, 3, 1, 2).to(torch.float32)
        convolutedFeatures = self.convolution(x)
        return convolutedFeatures.reshape([*shape[:-3], *convolutedFeatures.shape[-3:]])
