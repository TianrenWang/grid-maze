from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID

import numpy as np
import torch
import os
import argparse
import shutil
import csv
import uuid

from environments import FoggedMazeEnv
import models

parser = argparse.ArgumentParser()
parser.add_argument("--expName", type=str, default="default_exp")
args = parser.parse_args()


def generateLatents(mazeSize: int, modulePath: str, expName: str):
    env = FoggedMazeEnv(
        {
            "maze": None,
            "goal": (mazeSize // 2, mazeSize // 2),
            "start": None,
            "maxSteps": 1000,
            "mazeSize": mazeSize,
        }
    )
    module: models.LatentPathModule = RLModule.from_checkpoint(modulePath)
    obs, _ = env.reset()
    episodes = 0
    encounteredStates = set()
    latentStates = []
    stateLabels = []
    randomEpisodes = 0
    episodeRenders = dict()
    numRandomEpisodesToGenerate = 0

    while episodes < 200:
        gameId = str(uuid.uuid4())[:8]
        previousState = module.get_initial_state()
        obs, _ = env.reset()
        done = False
        episodicLatentStates = []
        episodicStateLabels = []
        wallEncounter = set()
        gotFinalStage = False
        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
            batched_obs = {
                Columns.OBS: obs,
                Columns.STATE_IN: {
                    k: torch.reshape(v, [1, -1]) for k, v in previousState.items()
                },
            }
            rl_module_out = module.forward_exploration(batched_obs)
            latent = (
                rl_module_out["actualLatents"].detach().cpu().numpy().flatten().tolist()
            )
            if env._agentLocation[0] <= 4:
                wallEncounter.add(0)
            if env._mazeSize - env._agentLocation[0] <= 4:
                wallEncounter.add(1)
            if env._agentLocation[1] <= 4:
                wallEncounter.add(2)
            if env._mazeSize - env._agentLocation[1] <= 4:
                wallEncounter.add(3)
            if str(latent) not in encounteredStates:
                goalDistance = np.abs(env._agentLocation - env._goalLocation)
                if goalDistance[0] <= 4 and goalDistance[1] <= 4:
                    latentStage = 5
                    if not gotFinalStage and len(episodicStateLabels):
                        episodicStateLabels[-1][2] = 4.5
                        gotFinalStage = True
                else:
                    latentStage = len(wallEncounter)
                encounteredStates.add(str(latent))
                episodicLatentStates.append(latent)
                numberOfDigitsInEpisodeLen = len(str(env._episode_len))
                if env._agentLocation[0] < 15 and env._agentLocation[1] < 15:
                    quadrant = 0
                elif env._agentLocation[0] >= 15 and env._agentLocation[1] >= 15:
                    quadrant = 3
                elif env._agentLocation[0] < 15:
                    quadrant = 1
                else:
                    quadrant = 2
                episodicStateLabels.append(
                    [
                        gameId,
                        env._episode_len,
                        latentStage,
                        f"{gameId}-{(3 - numberOfDigitsInEpisodeLen) * '0'}{env._episode_len}",
                        quadrant,
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
            if randomEpisodes < numRandomEpisodesToGenerate and env._episode_len < 200:
                action = np.random.choice(4)
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
            previousState = rl_module_out[Columns.STATE_OUT]

        episodeRenders[gameId] = env.render()
        if randomEpisodes < numRandomEpisodesToGenerate:
            print(gameId)
            print(episodeRenders[gameId])

        if env._episode_len < 100:
            episodes += 1
            for state in episodicLatentStates:
                latentStates.append(state)
            for label in episodicStateLabels:
                stateLabels.append(label)

    saveGameData(latentStates, stateLabels, expName)
    while True:
        gameId = input("Enter game ID: ")
        print(episodeRenders[gameId])


def saveGameData(
    states,
    stateLabels,
    dataName: str,
    columnNames=["game ID", "step", "stage", "positionId", "quadrant"],
):
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
