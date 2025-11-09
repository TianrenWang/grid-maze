import pickle
import os
import torch
import numpy as np
import argparse

from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


from maze import generateMaze, print_maze
from environments import MazeEnv, FoggedMazeEnv, ContinuousMazeEnv, SelfLocalizeEnv
from learners.ppo_grid_learner import PPOTorchLearnerWithSelfPredLoss
import models  # noqa: F401


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "True", "y")


parser = argparse.ArgumentParser()
parser.add_argument("--mazeSize", type=int, default=30)
parser.add_argument("--mazeName", type=str, default="default_maze")
parser.add_argument("--randomMaze", type=str2bool, default=True)
parser.add_argument("--hiddenSize", type=int, default=32)
parser.add_argument("--numLayers", type=int, default=2)
parser.add_argument("--maxSteps", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--expName", type=str, default="default_exp")
parser.add_argument("--numLearn", type=int, default=4000)
parser.add_argument("--evalInterval", type=int, default=100)
parser.add_argument("--fixedStart", type=str2bool, default=False)
parser.add_argument("--fogged", type=str2bool, default=True)
parser.add_argument("--grid", type=str2bool, default=False)
parser.add_argument("--selfLocalize", type=str2bool, default=False)
parser.add_argument("--memoryLen", type=int, default=20)
parser.add_argument("--debug", type=str2bool, default=False)
parser.add_argument("--gps", type=str2bool, default=False)
args = parser.parse_args()


def usesGrid():
    return args.grid or args.selfLocalize


if __name__ == "__main__":
    mazeSize = args.mazeSize
    mazeDimension = (mazeSize, mazeSize)
    goalLocation = (mazeSize // 2, mazeSize // 2)
    mazeName = args.mazeName
    mazesPath = "mazes"
    visionRange = 4

    if not args.randomMaze:
        if not os.path.exists(mazesPath):
            os.makedirs(mazesPath)
        mazes = os.listdir(mazesPath)

        if f"{mazeName}.pkl" in mazes:
            with open(f"{mazesPath}/{mazeName}.pkl", "rb") as file:
                maze = pickle.load(file)
        else:
            maze = generateMaze(mazeSize)
            with open(f"{mazesPath}/{mazeName}.pkl", "wb") as file:
                pickle.dump(maze, file)

        print_maze(maze)

    if usesGrid():
        module = models.ContinuousMazeModule
    elif args.gps:
        module = models.GPSModule
    elif args.memoryLen > 1 and args.fogged:
        module = models.MemoryMazeModule
    else:
        module = models.SimpleMazeModule

    if args.selfLocalize:
        env = SelfLocalizeEnv
    elif args.grid or args.gps:
        env = ContinuousMazeEnv
    elif args.fogged:
        env = FoggedMazeEnv
    else:
        env = MazeEnv

    environmentConfig = {
        "maze": None if args.randomMaze else maze,
        "goal": None if args.fixedStart else goalLocation,
        "start": [mazeSize // 2, mazeSize // 2] if args.fixedStart else None,
        "maxSteps": args.maxSteps,
        "memoryLen": args.memoryLen,
        "mazeSize": mazeSize,
        "debugging": args.debug,
    }

    agentConfig = (
        PPOConfig()
        .environment(env)
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=module,
                model_config={
                    "hiddenSize": args.hiddenSize,
                    "numLayers": args.numLayers,
                    "inputSize": visionRange * 2 + 1 if args.fogged else mazeSize,
                    "max_seq_len": args.memoryLen,
                    "mazeSize": mazeSize,
                },
            ),
        )
        .learners(num_gpus_per_learner=1 if torch.cuda.is_available() else 0)
        .evaluation(
            evaluation_num_env_runners=1 if args.debug else 8,
            evaluation_duration_unit="episodes",
            evaluation_duration=1 if args.debug else 24,
        )
        .training(
            lr=args.lr,
            entropy_coeff=0.01,
        )
    )
    if usesGrid():
        config = {"localization_coeff": 0.02, "self_localize": args.selfLocalize}
        agentConfig.training(
            learner_class=PPOTorchLearnerWithSelfPredLoss,
            learner_config_dict=config,
            lr=args.lr,
            entropy_coeff=0.01,
        )
    agentConfig.env_config = environmentConfig
    agent = agentConfig.build_algo()
    checkpointPath = f"{os.path.abspath(os.getcwd())}/checkpoints/{args.expName}"
    if os.path.exists(checkpointPath):
        agent.restore_from_path(checkpointPath)
    if args.debug:
        for i in range(10):
            agent.evaluate()
    else:
        numSamples = 0 if args.selfLocalize else 10
        for i in range(args.numLearn):
            result = agent.train()
            if i % args.evalInterval == 0:
                print(
                    f"Iteration {i + 1}",
                    " - ",
                    str(datetime.now())[:-7],
                )
                if args.selfLocalize:
                    predictionError = np.round(
                        result["learners"]["default_policy"]["prediction_error"], 2
                    )
                    print("Prediction Error:", predictionError)
                    positionError = np.round(
                        result["learners"]["default_policy"]["position_error"], 2
                    )
                    print("Position Error:", positionError)
                    placeBias = np.round(
                        result["learners"]["default_policy"]["place_bias"], 2
                    )
                    print("Place Bias:", placeBias)
                averageReturn = 0
                averageSteps = 0
                for j in range(numSamples):
                    result = agent.evaluate()["env_runners"]
                    averageReturn += result["episode_return_mean"]
                    averageSteps += result["episode_len_mean"]
                if not args.selfLocalize:
                    averageReturn = round(averageReturn / numSamples, 2)
                    averageSteps = round(averageSteps / numSamples, 0)
                    print("Steps:", averageSteps)
                    print("Performance:", averageReturn)
                    numSamples = int(10 * averageReturn / numSamples) + 1
                agent.save(checkpointPath)
