from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.core.rl_module.rl_module import RLModule
import numpy as np
import torch
from environments import FoggedMazeEnv
from maze import print_maze


def manualRun(mazeSize: int, module: RLModule, env: FoggedMazeEnv, envConfig):
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

    previousState = torch.zeros([1, module.linearHiddenSize], dtype=torch.float32)
    while not done and steps < 1000:
        batched_obs = {
            Columns.OBS: torch.from_numpy(np.array(env._memory)).unsqueeze(0),
            Columns.STATE_IN: {"h": previousState},
        }
        rl_module_out = module.forward_inference(batched_obs)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])[0, -1]
        action = np.random.choice(env.action_space.n, p=softmax(logits))
        previousMemoryLen = len(env._memory)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        steps += 1
        agentLocationValue = mazeTracker[env._agentLocation[0]][env._agentLocation[1]]
        if isinstance(agentLocationValue, int) and agentLocationValue < 9:
            mazeTracker[env._agentLocation[0]][env._agentLocation[1]] += 1
        totalReward += reward
        if previousMemoryLen == 10:
            previousState = rl_module_out[Columns.EMBEDDINGS][:, 0]

    for i in range(mazeSize):
        for j in range(mazeSize):
            if not mazeTracker[i][j]:
                mazeTracker[i][j] = " "

    print_maze(mazeTracker)
    print("Reward:", totalReward)
    print("Steps:", steps)
