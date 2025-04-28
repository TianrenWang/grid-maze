from ray.rllib.algorithms import Algorithm
import numpy as np
import torch
from environments import MazeEnv
from maze import print_maze


def manualRun(mazeSize: int, agent: Algorithm, env: MazeEnv, envConfig):
    env = env(envConfig)
    obs, _ = env.reset()
    policy = agent.get_policy()
    done = False
    steps = 0
    totalReward = 0
    maze = [[0 for i in range(mazeSize)] for i in range(mazeSize)]

    while not done and steps < 1000:
        batched_obs = {"vision": np.expand_dims(obs["vision"], axis=0)}
        _, _, info = policy.compute_actions(batched_obs, full_fetch=True)
        policyLogits = info["action_dist_inputs"][0]
        action = np.random.choice(
            [0, 1, 2, 3],
            size=1,
            p=torch.nn.functional.softmax(torch.Tensor(policyLogits), dim=0).numpy(),
        )[0]
        obs, reward, done, _, info = env.step(action)  # Step the environment
        steps += 1
        maze[env._agentLocation[0]][env._agentLocation[1]] += 1
        totalReward += reward
    print_maze(maze)
    print("Reward:", totalReward)
    print("Steps:", steps)
