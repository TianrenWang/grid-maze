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
    done = False
    steps = 0
    totalReward = 0
    maze = [[0 for i in range(mazeSize)] for i in range(mazeSize)]

    while not done and steps < 1000:
        batched_obs = {
            Columns.OBS: {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        }
        rl_module_out = module.forward_inference(batched_obs)
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
        action = np.random.choice(env.action_space.n, p=softmax(logits[0]))
        obs, reward, done, _, info = env.step(action)
        steps += 1
        agentLocationValue = maze[env._agentLocation[0]][env._agentLocation[1]]
        if agentLocationValue < 9:
            maze[env._agentLocation[0]][env._agentLocation[1]] += 1
        totalReward += reward
    print_maze(maze)
    print("Reward:", totalReward)
    print("Steps:", steps)
