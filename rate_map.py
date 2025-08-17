from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID
import matplotlib.pyplot as plt
import matplotlib
import random

matplotlib.use("Agg")

import numpy as np
import torch
import os
import argparse

from environments import PlaceMazeEnv
import models

parser = argparse.ArgumentParser()
parser.add_argument("--expName", type=str, default="default_exp")
args = parser.parse_args()


def getMoveByCoverage(env: PlaceMazeEnv, occupany: list[int]):
    bestMoveScore = 0
    bestMove = 0
    for action, direction in env._action_to_direction.items():
        newLocation = env._agentLocation + direction
        if env.isValidLocation(newLocation):
            newLocY = newLocation[0]
            newLocX = newLocation[1]
            occupancyOfArea = np.array(occupany)[newLocY, newLocX]
            jitter = random.uniform(0, 0.0001)
            coverage_score = 1.0 / (1 + occupancyOfArea) + jitter
            currentMoveScore = coverage_score
            if currentMoveScore > bestMoveScore:
                bestMove = action
                bestMoveScore = currentMoveScore
    return bestMove


def generateRatemap(mazeSize: int, modulePath: str, expName: str):
    env = PlaceMazeEnv(
        {
            "goal": goalLocation,
            "maxSteps": 1000,
            "memoryLen": 20,
            "mazeSize": mazeSize,
        }
    )
    module: models.PlaceMazeModule = RLModule.from_checkpoint(modulePath)
    obs, _ = env.reset()
    steps = 0
    occupancyCounter = []
    totalActivation = []
    for _ in range(mazeSize):
        currentRowCounter = []
        currentRowActivation = []
        occupancyCounter.append(currentRowCounter)
        totalActivation.append(currentRowActivation)
        for _ in range(mazeSize):
            currentRowCounter.append(0)
            currentRowActivation.append(np.zeros(module.gridSize))

    previousState = module.get_initial_state()
    while steps < 100000:
        obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
        batched_obs = {
            Columns.OBS: obs,
            Columns.STATE_IN: {
                k: torch.reshape(v, [1, -1]) for k, v in previousState.items()
            },
        }
        rl_module_out = module.forward_exploration(batched_obs)
        action = getMoveByCoverage(env, occupancyCounter)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        steps += 1
        occupancyCounter[env._agentLocation[0]][env._agentLocation[1]] += 1
        previousState = rl_module_out[Columns.STATE_OUT]
        decodedGrid = convert_to_numpy(
            module.gridDecoder(previousState["hiddenGrid"])
        ).flatten()
        currentActivation = totalActivation[env._agentLocation[0]][
            env._agentLocation[1]
        ]
        totalActivation[env._agentLocation[0]][env._agentLocation[1]] = (
            currentActivation + decodedGrid
        )
        if done or truncated:
            obs, _ = env.reset()
            previousState = module.get_initial_state()
    print("Finished simulations. Now plotting rate maps.")
    averageActivation = np.array(totalActivation) / np.reshape(
        np.array(occupancyCounter), [mazeSize, mazeSize, 1]
    )
    averageActivation = np.transpose(averageActivation, [2, 0, 1])
    averageActivation = np.reshape(averageActivation, [averageActivation.shape[0], -1])
    low = np.percentile(averageActivation, 1, 1)[:, None]
    high = np.percentile(averageActivation, 99, 1)[:, None]
    normalizedActivation = (np.clip(averageActivation, low, high) - low) / (high - low)
    normalizedActivation = np.reshape(
        normalizedActivation, [-1, env._mazeSize, env._mazeSize]
    )
    cellIndex = 0
    savePath = f"rate_maps/{expName}"
    os.makedirs(savePath, exist_ok=True)
    while cellIndex < len(normalizedActivation):
        fig, ax = plt.subplots()
        ax.imshow(
            normalizedActivation[cellIndex],
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        cellIndex += 1
        ax.axis("off")
        fig.savefig(
            f"{savePath}/{cellIndex}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)


if __name__ == "__main__":
    mazeSize = 30
    goalLocation = (mazeSize // 2, mazeSize // 2)
    checkpointPath = f"{os.path.abspath(os.getcwd())}/checkpoints/{args.expName}"
    rlModulePath = os.path.join(
        checkpointPath,
        "learner_group",
        "learner",
        "rl_module",
        DEFAULT_MODULE_ID,
    )
    generateRatemap(
        mazeSize,
        rlModulePath,
        args.expName,
    )
