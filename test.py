from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.core.rl_module.rl_module import RLModule
import numpy as np
import torch
from environments import MazeEnv
from maze import print_maze


def manualRun(mazeSize: int, module: RLModule, env: MazeEnv, envConfig):
    env = env(envConfig)
    obs, _ = env.reset()
    goalLocation = env._goalLocation
    done = False
    steps = 0
    totalReward = 0
    mazeTracker = env._mazeArray.copy()
    mazeTracker[mazeSize // 2][mazeSize // 2] = "S"
    mazeTracker[goalLocation[0]][goalLocation[1]] = "*"
    for i in range(mazeSize):
        for j in range(mazeSize):
            if not env._mazeArray[i][j]:
                mazeTracker[i][j] = "X"
            elif env._mazeArray[i][j] == 1:
                mazeTracker[i][j] = 0

    while not done and steps < 1000:
        batched_obs = {
            Columns.OBS: {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        }
        rl_module_out = module.forward_inference(batched_obs)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
        action = np.random.choice(env.action_space.n, p=softmax(logits[0]))
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        steps += 1
        agentLocationValue = mazeTracker[env._agentLocation[0]][env._agentLocation[1]]
        if isinstance(agentLocationValue, int) and agentLocationValue < 9:
            mazeTracker[env._agentLocation[0]][env._agentLocation[1]] += 1
        totalReward += reward

    for i in range(mazeSize):
        for j in range(mazeSize):
            if not env._mazeArray[i][j]:
                mazeTracker[i][j] = " "

    print_maze(mazeTracker)
    print("Reward:", totalReward)
    print("Steps:", steps)
