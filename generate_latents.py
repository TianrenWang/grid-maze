import argparse
import csv
import os
import shutil
import uuid

import matplotlib.pyplot as plt
import numpy as np
import torch
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

import models
from environments import SmoothExplorationEnv

parser = argparse.ArgumentParser()
parser.add_argument("--expName", type=str, default="default_exp")
parser.add_argument("--statistics", action="store_true")
args = parser.parse_args()


def group_similarity_matrix(groups):
    """
    groups = {
        "hash1": [vec1, vec2, ...],
        "hash2": [vec3, vec4, ...],
        ...
    }

    Returns:
        group_ids: list of group hashes
        matrix: NxN average cosine similarity matrix
    """

    group_ids = list(groups)

    # Normalize vectors
    normalized = {}
    for group_id, vectors in groups.items():
        X = np.asarray(vectors, dtype=float)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        normalized[group_id] = X

    n = len(group_ids)
    matrix = np.zeros((n, n))

    for i, group_a in enumerate(group_ids):
        for j, group_b in enumerate(group_ids):
            sims = normalized[group_a] @ normalized[group_b].T

            if i == j:
                # Exclude self-similarity (always 1.0)
                sims = sims[~np.eye(len(sims), dtype=bool)]

            matrix[i, j] = sims.mean()

    # Human-readable output
    print("Group similarity (average cosine similarity)")
    print()

    # Short numeric group labels
    labels = [str(g)[:8] for g in group_ids]

    print(f"{'':>10}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()

    for i, label in enumerate(labels):
        print(f"{label:>10}", end="")
        for j in range(n):
            print(f"{matrix[i, j]:10.3f}", end="")
        print()

    # totalSelfSimilarity = 0
    # totalAverageSimilarity = 0
    # totalMaxSimilarity = 0
    # totalMinSimilarity = 0
    # for i, label in enumerate(labels):
    #     totalSelfSimilarity += matrix[i, i]
    #     totalAverageSimilarity += matrix[i].mean()
    #     totalMaxSimilarity += matrix[i].max()
    #     totalMinSimilarity += matrix[i].min()
    # print("Self:", totalSelfSimilarity / len(labels))
    # print("Average:", totalAverageSimilarity / len(labels))
    # print("Max:", totalMaxSimilarity / len(labels))
    # print("Min:", totalMinSimilarity / len(labels))

    return group_ids, matrix


def generateLatents(
    mazeSize: int, modulePath: str, expName: str, statistics: bool = False
):
    env = SmoothExplorationEnv(
        {
            "maze": None,
            "start": None,
            "maxSteps": 30,
            "mazeSize": mazeSize,
        }
    )
    module: models.LatentPathModule = RLModule.from_checkpoint(modulePath)
    obs, _ = env.reset()
    episodes = 0
    encounteredStates = set()
    latentStates = []
    stateLabels = []
    positionInfos = {}
    latentsByPosition: dict[str, list[np.ndarray]] = {}
    positions: dict[str, tuple[int, int]] = {}

    while episodes < (2000 if statistics else 200):
        if episodes % 100 == 0 and statistics:
            print(episodes)
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
            rl_module_out = module.forward(batched_obs)
            latent = (
                rl_module_out["jepaLatent"].detach().cpu().numpy().flatten().tolist()
            )
            actionDistribution = (
                torch.softmax(
                    rl_module_out[Columns.ACTION_DIST_INPUTS].flatten(), dim=0
                )
                .detach()
                .cpu()
                .numpy()
            )

            if str(latent) not in encounteredStates:
                encounteredStates.add(str(latent))
                latentStates.append(latent)
                numberOfDigitsInEpisodeLen = len(str(env._episode_len))
                positionId = f"{gameId}-{(3 - numberOfDigitsInEpisodeLen) * '0'}{env._episode_len}"
                labels = [
                    gameId,
                    env._episode_len,
                    positionId,
                    env._agentLocation.tolist(),
                    np.abs(env._agentLocation - env._goalLocation).sum() < 1.01,
                    np.round(np.max(actionDistribution), 2),
                ]
                stateLabels.append(labels)
                positionInfos[positionId] = (
                    obs.flatten()[: 9**2 * 3].reshape(9, 9, 3).numpy()
                )
                positionString = str(env._agentLocation.tolist())
                if positionString not in latentsByPosition:
                    latentsByPosition[positionString] = [latent]
                    positions[positionString] = env._agentLocation.tolist()
                else:
                    latentsByPosition[positionString].append(latent)

            action = np.random.choice(4, p=actionDistribution)
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
            previousState = rl_module_out[Columns.STATE_OUT]

        episodes += 1

    if statistics:
        latentsByPositionTrue = {}
        positionsTrue = {}
        for key, latents in latentsByPosition.items():
            if len(latents) > 30 and len(latentsByPositionTrue) < 17:
                latentsByPositionTrue[key] = latents
                positionsTrue[key] = positions[key]
        group_similarity_matrix(latentsByPositionTrue)
    else:
        saveGameData(latentStates, stateLabels, expName)
        while True:
            pos1Id = input("Pos1 ID: ")
            plt.imshow(positionInfos[pos1Id])
            plt.axis("off")
            plt.show()


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
            "done",
            "confidence",
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
        args.statistics,
    )
