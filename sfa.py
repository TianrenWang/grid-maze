import argparse
import os

import numpy as np
import torch
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

import models
from environments import PlaceMazeEnv

parser = argparse.ArgumentParser()
parser.add_argument("--expName", type=str, default="default_exp")
args = parser.parse_args()


def linear_sfa(latents: torch.Tensor, mask: torch.Tensor, num_components=2, eps=1e-3):
    """
    Linear Slow Feature Analysis with sequence masking.

    Args:
        latents: Tensor of shape (B, T, D)
        mask: Bool Tensor of shape (B, T), True = valid timestep
        num_components: number of slow features
        eps: numerical stability

    Returns:
        slow_features: Tensor of shape (B, T, num_components)
        W: projection matrix (D, num_components)
    """

    _, _, D = latents.shape

    # -----------------------
    # State covariance Cx
    # -----------------------
    x = latents[mask]  # (N, D)

    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean

    Cx = (x_centered.T @ x_centered) / (x_centered.shape[0] - 1)

    # -----------------------
    # Derivative covariance Cdx
    # Only keep valid transitions
    # -----------------------
    transition_mask = mask[:, :-1] & mask[:, 1:]

    dx = torch.diff(latents, dim=1)
    dx = dx[transition_mask]  # (N-1, D)

    dx_mean = dx.mean(dim=0, keepdim=True)
    dx_centered = dx - dx_mean

    Cdx = (dx_centered.T @ dx_centered) / (dx_centered.shape[0] - 1)

    # -----------------------
    # Generalized eigenproblem
    # Cdx w = lambda Cx w
    # -----------------------
    Cx = Cx + eps * torch.eye(D, device=latents.device)

    L = torch.linalg.cholesky(Cx)
    Linv = torch.linalg.inv(L)

    M = Linv @ Cdx @ Linv.T

    eigvals, eigvecs = torch.linalg.eigh(M)

    # smallest eigenvalues = slowest
    idx = torch.argsort(eigvals)[:num_components]

    W = Linv.T @ eigvecs[:, idx]

    # -----------------------
    # Project all latents
    # Keep original shape
    # -----------------------
    slow_features = (latents - x_mean) @ W

    # Optional: zero invalid positions
    slow_features = slow_features * mask.unsqueeze(-1)

    return slow_features, W


def generateLatents(mazeSize: int, modulePath: str):
    env = PlaceMazeEnv(
        {
            "maze": None,
            "maxSteps": 100,
            "mazeSize": mazeSize,
        }
    )
    module: models.LatentPathModule = RLModule.from_checkpoint(modulePath)
    obs, _ = env.reset()
    episodes = 0
    latentStates = []

    while episodes < 2000:
        previousState = module.get_initial_state()
        obs, _ = env.reset()
        done = False
        episodicLatentStates = []
        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
            batched_obs = {
                Columns.OBS: obs,
                Columns.STATE_IN: {
                    k: torch.reshape(v, [1, -1]) for k, v in previousState.items()
                },
            }
            rl_module_out = module.forward(batched_obs)
            latent = rl_module_out[Columns.STATE_OUT]["hiddenObs"].detach().cpu()
            episodicLatentStates.append(latent.unsqueeze(1))

            action = np.random.choice(
                4,
                p=torch.softmax(
                    rl_module_out[Columns.ACTION_DIST_INPUTS].flatten(), dim=0
                )
                .detach()
                .cpu()
                .numpy(),
            )
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
            previousState = rl_module_out[Columns.STATE_OUT]

        episodes += 1
        latentStates.append(torch.concat(episodicLatentStates, dim=1).squeeze(0))

    paddedLatents = torch.nn.utils.rnn.pad_sequence(latentStates, batch_first=True)
    lengths = torch.tensor([len(x) for x in latentStates])
    mask = torch.arange(paddedLatents.size(1))[None, :] < lengths[:, None]
    return paddedLatents, mask


if __name__ == "__main__":
    mazeSize = 30
    checkpointPath = f"{os.path.abspath(os.getcwd())}/checkpoints/{args.expName}"
    rlModulePath = os.path.join(
        checkpointPath,
        "learner_group",
        "learner",
        "rl_module",
        DEFAULT_MODULE_ID,
    )
    latents, mask = generateLatents(mazeSize, rlModulePath)
    slowFeatures, _ = linear_sfa(latents, mask, 2)
    showCounter = 0
    for episode in slowFeatures:
        lastState = episode[0]
        finished = False
        for state in episode:
            if state.sum().item() == 0:
                finished = True
                break
            lastState = state
        if finished:
            print(torch.round(lastState, decimals=2))
            showCounter += 1
        if showCounter > 20:
            break
