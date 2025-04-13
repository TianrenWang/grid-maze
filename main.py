import pickle
import os
import torch
import numpy as np

from datetime import datetime
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from maze import generate_maze, print_maze
from environment import GridMazeEnv
import models  # noqa: F401


if __name__ == "__main__":
    mazeSize = 7
    centerPosition = (mazeSize - 1) // 2
    mazeDimension = (mazeSize, mazeSize)
    goalLocation = (centerPosition, centerPosition)
    id = f"{mazeSize}by{mazeSize}"
    mazesPath = "mazes"
    debug = False

    if not os.path.exists(mazesPath):
        os.makedirs(mazesPath)
    mazes = os.listdir(mazesPath)

    if f"{id}.pkl" in mazes:
        with open(f"{mazesPath}/{id}.pkl", "rb") as file:
            maze = pickle.load(file)
    else:
        maze = generate_maze(mazeDimension, goalLocation)
        with open(f"{mazesPath}/{id}.pkl", "wb") as file:
            pickle.dump(maze, file)

    print_maze(maze)

    agentConfig = (
        PPOConfig()
        .environment(GridMazeEnv)
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .training(
            model={
                "custom_model": "simple_maze_net",
                "custom_model_config": {
                    "hiddenSize": 16,
                    "numLayers": 5,
                    "mazeSize": mazeSize,
                },
            },
            lr=1e-6,
            entropy_coeff=0.01,
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .evaluation(
            evaluation_interval=10,
            evaluation_num_env_runners=8,
            evaluation_duration_unit="episodes",
            evaluation_duration=256,
        )
    )
    environmentConfig = {
        "maze": maze,
        "goal": [centerPosition] * 2,
        "start": [3, 1],
        "maxSteps": 1000,
    }
    agentConfig.env_config = environmentConfig
    agent = agentConfig.build_algo()
    checkpointPath = f"{os.path.abspath(os.getcwd())}/checkpoints/{id}"
    if os.path.exists(checkpointPath):
        agent.restore(checkpointPath)

    def manualRun(agent: Algorithm):
        env = GridMazeEnv(environmentConfig)
        obs, _ = env.reset()
        policy = agent.get_policy()
        done = False
        steps = 0

        while not done and steps < 10:
            _, _, info = policy.compute_actions([obs], full_fetch=True)
            policyLogits = info["action_dist_inputs"][0]
            print(
                "Action logits:",
                torch.nn.functional.softmax(torch.Tensor(policyLogits)),
            )
            action = np.argmax(policyLogits)
            obs, reward, done, _, info = env.step(action)  # Step the environment
            steps += 1

    for i in range(2000):
        result = agent.train()
        if "evaluation" in result:
            print(
                f"Iteration {i + 1}:",
                np.rint(result["evaluation"]["env_runners"]["episode_reward_mean"]),
                " - ",
                datetime.now(),
            )
            if debug:
                manualRun(agent)
            agent.save(checkpointPath)
