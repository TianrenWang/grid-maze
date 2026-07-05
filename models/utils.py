import torch
import math


def calculatePlace(
    placeCells: torch.Tensor,
    agentLocation: torch.Tensor,
    numPlaceCells: int = 32,
    fieldSize: float = 0.3 / math.sqrt(32),
):
    with torch.no_grad():
        agentLocationShape = agentLocation.shape
        agentLocation = agentLocation.flatten(0, -2)
        diff = agentLocation.unsqueeze(1) - placeCells.unsqueeze(0)
        dists_squared = torch.sum(torch.abs(diff) ** 2, dim=-1)
        unnormalized_activations = -dists_squared / (2 * fieldSize)
        normalized_activations = torch.nn.functional.softmax(
            unnormalized_activations, dim=1
        )
        return normalized_activations.reshape([*agentLocationShape[:2], numPlaceCells])
