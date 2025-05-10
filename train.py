import pickle
import os
import torch
import numpy as np
import argparse

from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec, RLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core import DEFAULT_MODULE_ID


from maze import generate_maze, print_maze
from environments import MazeEnv, FoggedMazeEnv
from test import manualRun
import models  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--mazeSize", type=int, default=19)
parser.add_argument("--mazeName", type=str, default="default_maze")
parser.add_argument("--hiddenSize", type=int, default=32)
parser.add_argument("--numLayers", type=int, default=2)
parser.add_argument("--maxSteps", type=int, default=5000)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--expName", type=str, default="default_exp")
parser.add_argument("--numLearn", type=int, default=2000)
parser.add_argument("--evalInterval", type=int, default=100)
parser.add_argument("--fixedStart", type=bool, default=True)
parser.add_argument("--fogged", type=bool, default=False)
parser.add_argument("--placeCells", type=bool, default=False)
parser.add_argument("--memoryLen", type=int, default=0)
parser.add_argument("--gateCloseRate", type=float, default=0)
args = parser.parse_args()


if __name__ == "__main__":
    mazeSize = args.mazeSize
    mazeDimension = (mazeSize, mazeSize)
    goalLocation = (mazeSize - 2, mazeSize - 2)
    mazeName = args.mazeName
    mazesPath = "mazes"
    visionRange = 4

    if not os.path.exists(mazesPath):
        os.makedirs(mazesPath)
    mazes = os.listdir(mazesPath)

    if f"{mazeName}.pkl" in mazes:
        with open(f"{mazesPath}/{mazeName}.pkl", "rb") as file:
            maze = pickle.load(file)
    else:
        maze = generate_maze(mazeDimension, goalLocation)
        with open(f"{mazesPath}/{mazeName}.pkl", "wb") as file:
            pickle.dump(maze, file)

    print_maze(maze)

    if args.placeCells:
        module = models.PlaceMazeModule
    elif args.memoryLen > 1 and args.fogged:
        module = models.MemoryMazeModule
    else:
        module = models.SimpleMazeModule

    env = FoggedMazeEnv if args.fogged else MazeEnv

    agentConfig = (
        PPOConfig()
        .environment(env)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        module_class=module,
                        model_config={
                            "hiddenSize": args.hiddenSize,
                            "numLayers": args.numLayers,
                            "inputSize": visionRange * 2 + 1
                            if args.fogged
                            else mazeSize,
                        },
                    ),
                }
            ),
        )
        .learners(num_gpus_per_learner=1 if torch.cuda.is_available() else 0)
        .training(
            lr=args.lr,
            entropy_coeff=0.01,
        )
        .evaluation(
            evaluation_interval=args.evalInterval,
            evaluation_num_env_runners=8,
            evaluation_duration_unit="episodes",
            evaluation_duration=256,
        )
    )
    environmentConfig = {
        "maze": maze,
        "goal": list(goalLocation),
        "start": [1, 1] if args.fixedStart else None,
        "maxSteps": args.maxSteps,
        "memoryLen": args.memoryLen,
        "gateCloseRate": args.gateCloseRate,
    }
    agentConfig.env_config = environmentConfig
    agent = agentConfig.build_algo()
    checkpointPath = f"{os.path.abspath(os.getcwd())}/checkpoints/{args.expName}"
    if os.path.exists(checkpointPath):
        agent.restore(checkpointPath)

    for i in range(args.numLearn):
        result = agent.train()
        if i % args.evalInterval == 0:
            print(
                f"Iteration {i + 1}:",
                np.rint(result["evaluation"]["env_runners"]["episode_return_mean"]),
                " - ",
                datetime.now(),
            )
            agent.save(checkpointPath)
            module = RLModule.from_checkpoint(
                os.path.join(
                    checkpointPath,
                    "learner_group",
                    "learner",
                    "rl_module",
                    DEFAULT_MODULE_ID,
                )
            )
            manualRun(mazeSize, module, env, environmentConfig)
