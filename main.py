import pickle
import os

from ray.rllib.algorithms.ppo import PPOConfig
from maze import generate_maze
from environment import GridMazeEnv
import models  # noqa: F401


if __name__ == "__main__":
    mazeSize = 5
    centerPosition = (mazeSize - 1) // 2
    mazeDimension = (mazeSize, mazeSize)
    goalLocation = (centerPosition, centerPosition)
    id = "5by5"

    mazes = os.listdir("maze")

    if id in mazes:
        with open(f"maze/{id}.pkl", "r") as file:
            maze = pickle.load(file)
    else:
        maze = generate_maze(mazeDimension, goalLocation)
        with open(f"maze/{id}.pkl", "wb") as file:
            pickle.dump(maze, file)

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
    )

    agentConfig.env_config = {"maze": maze, "goal": [centerPosition] * 2}
    agent = agentConfig.build_algo()
    checkpointPath = f"checkpoints/{id}/"
    if os.path.exists(checkpointPath):
        agent.restore(checkpointPath)
    agent.train()
    checkpoint_path = agent.save(checkpointPath)
