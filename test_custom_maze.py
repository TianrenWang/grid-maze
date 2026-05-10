from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import os
import argparse

from environments import FoggedMazeEnv
import models

customMaze = [[0 if j == 4 else 1 for j in range(30)] for i in range(30)]
customMaze[26][4] = 1
mazeSize = 30

# customMaze = generateMaze(mazeSize)
# for row in customMaze:
#     for i in range(len(row)):
#         if i < 4:
#             row[i] = 0
#     row[4] = 0

# customMaze[26] = [1 for i in range(mazeSize)]

parser = argparse.ArgumentParser()
parser.add_argument("--expName", type=str, default="default_exp")
args = parser.parse_args()


if __name__ == "__main__":
    agentConfig = (
        PPOConfig()
        .environment(FoggedMazeEnv)
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=models.LatentPathModule,
                model_config={"hiddenSize": 32, "inputSize": 9, "max_seq_len": 20},
            ),
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_duration_unit="episodes",
            evaluation_duration=1,
        )
    )
    agentConfig.env_config = {
        "maze": customMaze,
        "goal": (mazeSize // 2, mazeSize // 2),
        "start": (0, 0),
        "maxSteps": 1000,
        "mazeSize": mazeSize,
        "debugging": True,
    }
    agent = agentConfig.build_algo()
    checkpointPath = f"{os.path.abspath(os.getcwd())}/checkpoints/{args.expName}"
    if os.path.exists(checkpointPath):
        agent.restore_from_path(checkpointPath)
    agent.evaluate()
