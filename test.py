from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.core.rl_module.rl_module import RLModule
import random
import numpy as np
import torch
from environments import FoggedMazeEnv, PlaceMazeEnv
from maze import print_maze


def manualRun(
    mazeSize: int, module: RLModule, env: FoggedMazeEnv | PlaceMazeEnv, envConfig
):
    env = env(envConfig)
    obs, _ = env.reset()
    goalLocation = env._goalLocation
    done = False
    steps = 0
    totalReward = 0
    mazeTracker = []
    for i in range(mazeSize):
        currentRow = []
        mazeTracker.append(currentRow)
        for j in range(mazeSize):
            originalValue = env._mazeArray[i][j]
            if not originalValue:
                currentRow.append("X")
            elif originalValue == 1:
                currentRow.append(0)
            else:
                currentRow.append(originalValue)
    mazeTracker[mazeSize // 2][mazeSize // 2] = "S"
    mazeTracker[goalLocation[0]][goalLocation[1]] = "*"

    previousState = module.get_initial_state()
    actualObs = None
    while not done and steps < 1000:
        obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
        if random.random() < 1 / envConfig["memoryLen"] or actualObs is None:
            actualObs = obs
        else:
            actualObs = torch.concat([actualObs, obs], dim=1)
        batched_obs = {
            Columns.OBS: actualObs,
            Columns.STATE_IN: {
                k: torch.reshape(v, [1, -1]) for k, v in previousState.items()
            },
        }
        rl_module_out = module.forward_inference(batched_obs)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])[0, -1]
        action = np.random.choice(env.action_space.n, p=softmax(logits))
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        steps += 1
        agentLocationValue = mazeTracker[env._agentLocation[0]][env._agentLocation[1]]
        if isinstance(agentLocationValue, int) and agentLocationValue < 9:
            mazeTracker[env._agentLocation[0]][env._agentLocation[1]] += 1
        totalReward += reward
        previousState = rl_module_out[Columns.STATE_OUT]

    for i in range(mazeSize):
        for j in range(mazeSize):
            if not mazeTracker[i][j]:
                mazeTracker[i][j] = " "

    print_maze(mazeTracker)
    print("Reward:", totalReward)
    print("Steps:", steps)
