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
        self.memory = nn.GRU(actionSize, latentSize, batch_first=True)
        self.predictor = nn.Linear(latentSize, latentSize)
        self.EMAEncoder = EMA(
            self.encoder,
            beta=0.9999,
            update_after_step=0,
            update_every=10,
            min_value=0.9999,
        )

    def _getContext(self, vision: torch.Tensor):
        return self.encoder(vision[:, 0, :, :, :].unsqueeze(1)).squeeze(1)

    def forward(self, vision: torch.Tensor, action: torch.Tensor):
        contextLatent = self._getContext(vision)
        memory, _ = self.memory(action, contextLatent.unsqueeze(0))
        return memory

    def forward_train(self, vision: torch.Tensor, action: torch.Tensor):
        contextLatent = self._getContext(vision)
        memory, _ = self.memory(action, contextLatent.unsqueeze(0))
        predictedLatent = self.predictor(memory)
        targetLatent = self.EMAEncoder(vision).detach()
        predictionLoss = torch.mean((predictedLatent - targetLatent) ** 2, dim=-1)

        return memory, predictionLoss, targetLatent
