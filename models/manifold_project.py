import math

import torch
from ema_pytorch import EMA
from torch import nn


class ManifoldProjector(nn.Module):
    def __init__(self, latentSize: int, actionSize: int, manifoldDim: int):
        super().__init__()
        self.manifoldProjector = nn.Sequential(
            nn.Linear(latentSize, latentSize),
            nn.ReLU(),
            nn.Linear(latentSize, manifoldDim),
            nn.Sigmoid(),
        )
        self.manifoldCoordinateReadout = nn.Sequential(
            nn.Linear(manifoldDim, manifoldDim),
            nn.ReLU(),
            nn.Linear(manifoldDim, manifoldDim),
            nn.Sigmoid(),
        )
        self.EMAProjector = EMA(
            self.manifoldProjector,
            beta=0.9999,
            update_after_step=0,
            update_every=10,
            min_value=0.9999,
        )
        self.directionDecoder = nn.Sequential(nn.Linear(actionSize, 1), nn.Tanh())
        self.speed = 1 / 31

    def _getDisplacement(self, latent: torch.Tensor) -> torch.Tensor:
        angle = self.directionDecoder(latent) * math.pi
        return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1) * self.speed

    def forward(self, latent: torch.Tensor):
        return self.EMAProjector(latent)

    def forward_train(self, latent: torch.Tensor, action: torch.Tensor):
        startingPlace = self.manifoldProjector(latent[:, 0])
        displacement = self._getDisplacement(action)
        firstDisplacement = displacement[:, 0, :]
        displacement[:, 0, :] = torch.zeros(
            firstDisplacement.shape,
            dtype=torch.float32,
            device=firstDisplacement.device,
        )
        predictedCoordinates = startingPlace.unsqueeze(1) + torch.cumsum(
            displacement, dim=1
        )
        return predictedCoordinates, self.manifoldCoordinateReadout(
            predictedCoordinates.detach()
        )
