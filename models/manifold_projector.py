import math

import torch
from ema_pytorch import EMA
from torch import nn


class ManifoldProjector(nn.Module):
    def __init__(self, latentSize: int, actionSize: int, manifoldDim: int):
        super().__init__()
        self.manifoldProjector = nn.Sequential(
            nn.Linear(latentSize, latentSize),
            nn.Tanh(),
            nn.Linear(latentSize, latentSize),
            nn.Tanh(),
            nn.Linear(latentSize, latentSize),
            nn.Tanh(),
            nn.Linear(latentSize, manifoldDim),
        )
        self.EMAProjector = EMA(
            self.manifoldProjector,
            beta=0.99,
            update_after_step=0,
            update_every=10,
            min_value=0.99,
        )
        self.directionDecoder = nn.Sequential(nn.Linear(actionSize, 1), nn.Tanh())
        self.speed = 1 / 31

    def _getDisplacement(self, latent: torch.Tensor) -> torch.Tensor:
        angle = self.directionDecoder(latent) * math.pi
        return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1) * self.speed

    def _getCrossScore(self):
        directions = self._getDisplacement(
            torch.tensor(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ],
                dtype=torch.float32,
                device=next(self.manifoldProjector.parameters()).device,
            )
        )
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        theta = torch.atan2(directions[:, 1], directions[:, 0])
        z = torch.stack(
            [
                torch.cos(2 * theta),
                torch.sin(2 * theta),
            ],
            dim=-1,
        )
        magnitude = z.sum(dim=-2).norm(dim=-1)
        score = 1 - magnitude / 4
        return score

    def forward(self, latent: torch.Tensor):
        return self.EMAProjector(latent)

    def forward_train(self, latent: torch.Tensor, action: torch.Tensor):
        startingPlace = self.EMAProjector(latent[:, 0])
        displacement = self._getDisplacement(action)
        firstDisplacement = displacement[:, 0, :]
        displacement[:, 0, :] = torch.zeros(
            firstDisplacement.shape,
            dtype=torch.float32,
            device=firstDisplacement.device,
        )
        targetCoordinates = startingPlace.unsqueeze(1) + torch.cumsum(
            displacement, dim=1
        )
        predictedCoordinates = self.manifoldProjector(latent)
        return predictedCoordinates, targetCoordinates
