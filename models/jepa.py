import torch
from ema_pytorch import EMA
from torch import nn

from .simple_conv import SimpleConv


class Encoder(nn.Module):
    def __init__(self, inputSize: int, latentSize: int):
        super().__init__()
        self.visionEncoder = SimpleConv(latentSize)
        visionEncoderOutSize = ((inputSize + 1) // 2 + 1) // 2
        self.encoder = nn.Linear(
            visionEncoderOutSize**2 * latentSize * 2,
            latentSize,
        )

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        visionFeatures = self.visionEncoder(vision).flatten(2)
        return self.encoder(visionFeatures)


class JEPA(nn.Module):
    def __init__(self, inputSize: int, latentSize: int, actionSize: int):
        super().__init__()
        self.encoder = Encoder(inputSize, latentSize)
        self.memory = nn.GRU(latentSize, latentSize, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(latentSize + actionSize, latentSize),
            nn.ReLU(),
            nn.Linear(latentSize, latentSize),
        )
        self.EMAEncoder = EMA(
            self.encoder, beta=0.9999, update_after_step=100, update_every=10
        )

    def forward(self, vision: torch.Tensor, initialMemory: torch.Tensor):
        encodedVision = self.visionEncoder(vision)
        memory, _ = self.memory(encodedVision, initialMemory)
        return memory

    def forward_train(
        self, vision: torch.Tensor, initialMemory: torch.Tensor, action: torch.Tensor
    ):
        contextLatent = self.encoder(vision)
        memory, _ = self.memory(contextLatent, initialMemory.unsqueeze(0))
        predictedLatent = self.predictor(torch.concat([memory, action], dim=2))[
            :, :-1, :
        ]
        targetLatent = self.EMAEncoder(vision).detach()[:, 1:, :]
        predictionLoss = torch.sum((predictedLatent - targetLatent) ** 2, dim=-1)
        predictionLoss = torch.nn.functional.pad(predictionLoss, (0, 1))

        return memory, predictionLoss
