import pickle
import os
import torch

from datetime import datetime
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
                "custom_model": "resnet",
                "custom_model_config": {
                    "hiddenSize": 16,
                    "numLayers": 3,
                    "mazeSize": mazeSize,
                },
            }
        )
        .resources(num_gpus=1 if torch.cuda.is_available() else 0)
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=8,
            evaluation_duration_unit="episodes",
            evaluation_duration=16,
        )
    )

    agentConfig.env_config = {
        "maze": maze,
        "goal": [centerPosition] * 2,
        # "start": [5, 1],
        "maxSteps": 1000,
    }
    agent = agentConfig.build_algo()
    checkpointPath = f"checkpoints/{id}/"
    # if os.path.exists(checkpointPath):
    #     agent.restore(checkpointPath)

    print(datetime.now())
    for i in range(10):
        result = agent.train()
        print(result["evaluation"]["env_runners"]["episode_reward_mean"])
        print(datetime.now())
    # checkpoint_path = agent.save(checkpointPath)
