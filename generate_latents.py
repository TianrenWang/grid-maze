import argparse
import csv
import os
import shutil
import uuid

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


def generateLatents(mazeSize: int, modulePath: str, expName: str):
    env = PlaceMazeEnv(
        {
            "maze": None,
            "start": None,
            "maxSteps": 40,
            "mazeSize": mazeSize,
        }
    )
    module: models.LatentPathModule = RLModule.from_checkpoint(modulePath)
    obs, _ = env.reset()
    episodes = 0
    encounteredStates = set()
    latentStates = []
    stateLabels = []

    while episodes < 200:
        gameId = str(uuid.uuid4())[:8]
        previousState = module.get_initial_state()
        obs, _ = env.reset()
        done = False
        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
            batched_obs = {
                Columns.OBS: obs,
                Columns.STATE_IN: {
                    k: torch.reshape(v, [1, -1]) for k, v in previousState.items()
                },
            }
            rl_module_out = module.forward_exploration(batched_obs)
            latent = rl_module_out["latent"].detach().cpu().numpy().flatten().tolist()
            projection = (
                rl_module_out["projection"].detach().cpu().numpy().flatten().tolist()
            )

            if str(latent) not in encounteredStates:
                encounteredStates.add(str(latent))
                latentStates.append(latent)
                numberOfDigitsInEpisodeLen = len(str(env._episode_len))
                stateLabels.append(
                    [
                        gameId,
                        env._episode_len,
                        f"{gameId}-{(3 - numberOfDigitsInEpisodeLen) * '0'}{env._episode_len}",
                        np.round(env._agentLocation, decimals=2).tolist(),
                        np.round(projection, decimals=2).tolist(),
                        np.round(
                            (env._agentLocation - env._goalLocation).sum(), decimals=2
                        )
                        < 1.01,
                    ]
                )
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

    saveGameData(latentStates, stateLabels, expName)


def saveGameData(
    states,
    stateLabels,
    dataName: str,
    columnNames=None,
):
    if columnNames is None:
        columnNames = [
            "game ID",
            "step",
            "positionId",
            "location",
            "projection",
            "done",
        ]

    folder_path = "data/" + dataName
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    with open(folder_path + "/states.tsv", "a", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(states)

    hasFirstRow = False
    stateLabelsFilePath = folder_path + "/stateLabels.tsv"
    if os.path.exists(stateLabelsFilePath):
        with open(stateLabelsFilePath, "r") as file:
            firstLine = file.readline()
            hasFirstRow = not firstLine.strip()
    else:
        hasFirstRow = True

    with open(stateLabelsFilePath, "a", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        if hasFirstRow:
            writer.writerow(columnNames)
        writer.writerows(stateLabels)


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
    generateLatents(
        mazeSize,
        rlModulePath,
        args.expName,
    )
