import torch


def calculatePlace(
    placeCells: torch.Tensor,
    agentLocation: torch.Tensor,
    fieldSize: float = 0.06,
):
    numPlaceCells = placeCells.shape[0]
    agentLocationShape = agentLocation.shape
    agentLocation = agentLocation.flatten(0, -2)
    diff = agentLocation.unsqueeze(1) - placeCells.unsqueeze(0)
    dists_squared = torch.sum(torch.abs(diff) ** 2, dim=-1)
    unnormalized_activations = -dists_squared / (2 * fieldSize**2)
    normalized_activations = torch.nn.functional.softmax(
        unnormalized_activations, dim=1
    )
    return normalized_activations.reshape([*agentLocationShape[:-1], numPlaceCells])
