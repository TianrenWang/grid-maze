import argparse
import logging
import os
import pickle
from datetime import datetime

logging.getLogger("ray.rllib.algorithms.ppo.torch.ppo_torch_learner").setLevel(
    logging.ERROR
)

import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import models
from environments import MazeEnv, PlaceMazeEnv, SmoothExplorationEnv
from learners.ppo_grid_learner import PPOTorchLearnerWithSelfPredLoss
from maze import generateMaze, getMazeDebugString

parser = argparse.ArgumentParser()
parser.add_argument("--mazeSize", type=int, default=31)
parser.add_argument("--mazeName", type=str, default="default_maze")
parser.add_argument("--staticMaze", action="store_false", dest="randomMaze")
parser.add_argument("--hiddenSize", type=int, default=32)
parser.add_argument("--numLayers", type=int, default=2)
parser.add_argument("--maxSteps", type=int, default=58)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--expName", type=str, default="default_exp")
parser.add_argument("--numLearn", type=int, default=4000)
parser.add_argument("--evalInterval", type=int, default=100)
parser.add_argument("--fixedStart", action="store_true")
parser.add_argument("--noFog", action="store_false", dest="fogged")
parser.add_argument("--grid", action="store_true")
parser.add_argument("--selfLocalize", action="store_true")
parser.add_argument("--memoryLen", type=int, default=20)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--gps", action="store_true")
parser.add_argument("--latentPath", action="store_true")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--pureCode", action="store_true")
parser.add_argument("--entropy", type=float, default=0.1)
parser.add_argument("--visionPolicy", action="store_true")
parser.add_argument("--learnManifold", action="store_true")
args = parser.parse_args()

if args.pretrain or args.learnManifold:
    args.latentPath = True

if args.learnManifold:
    args.selfLocalize = True


def usesGrid():
    return args.grid or args.selfLocalize or args.latentPath or args.pureCode


if __name__ == "__main__":
    mazeSize = args.mazeSize
    mazeName = args.mazeName
    mazesPath = "mazes"
    visionRange = 4
    maze = None
    evalMaxSteps = 200

    if args.selfLocalize:
        maze = generateMaze(mazeSize)
    elif not args.randomMaze:
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

        getMazeDebugString(maze)

    if usesGrid():
        module = models.PathIntegrationWithVisionModule
        if args.latentPath:
            module = models.LatentPathModule
        elif args.pureCode:
            module = models.PureCodeModule
    elif args.gps:
        module = models.GPSModule
    elif args.memoryLen > 1 and args.fogged:
        module = models.PlaceCellControlModule
    else:
        module = models.SimpleMazeModule

    if args.selfLocalize:
        env = SmoothExplorationEnv
    elif usesGrid() or args.gps or args.fogged:
        env = PlaceMazeEnv
    else:
        env = MazeEnv

    environmentConfig = {
        "maze": maze if maze else None,
        "start": [mazeSize // 2, mazeSize // 2] if args.fixedStart else None,
        "maxSteps": args.maxSteps,
        "memoryLen": args.memoryLen,
        "mazeSize": mazeSize,
        "debugging": args.debug,
    }

    environmentEvalConfig = environmentConfig.copy()
    environmentEvalConfig["eval"] = True
    environmentEvalConfig["maxSteps"] = evalMaxSteps

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
                    "self_localize": args.selfLocalize,
                    "pretrain": args.pretrain,
                    "visionPolicy": args.visionPolicy,
                    "learnManifold": args.learnManifold,
                },
            ),
        )
        .learners(num_gpus_per_learner=1 if torch.cuda.is_available() else 0)
        .evaluation(
            evaluation_num_env_runners=1 if args.debug else 8,
            evaluation_duration_unit="episodes",
            evaluation_duration=1 if args.debug else 128,
            evaluation_config={"env_config": environmentEvalConfig},
        )
        .training(
            lr=args.lr,
            entropy_coeff=args.entropy,
        )
    )
    if usesGrid():
        config = {"localization_coeff": 0.02, "self_localize": args.selfLocalize}
        agentConfig.training(
            learner_class=PPOTorchLearnerWithSelfPredLoss,
            learner_config_dict=config,
            lr=args.lr,
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
        numSamples = 1
        for i in range(args.numLearn):
            result = agent.train()
            if i == 0 or (i + 1) % args.evalInterval == 0:
                print(
                    f"Iteration {i + 1}",
                    " - ",
                    str(datetime.now())[:-7],  # noqa: DTZ005
                )
                if args.selfLocalize or args.learnManifold:
                    trainingOutputs = result["learners"]["default_policy"]
                    if "prediction_error" in trainingOutputs:
                        predictionError = np.round(
                            trainingOutputs["prediction_error"], 2
                        )
                        print("Prediction Error:", predictionError)
                        positionError = np.round(trainingOutputs["position_error"], 2)
                        print("Position Error:", positionError)

                    if "jepa_loss" in trainingOutputs:
                        jepaLoss = np.round(trainingOutputs["jepa_loss"], 2)
                        print("JEPA Loss:", jepaLoss)

                    if "movement_loss" in trainingOutputs:
                        movementLoss = np.round(trainingOutputs["movement_loss"], 2)
                        print("Movement Loss:", movementLoss)
                else:
                    averageReturn = 0
                    averageSteps = 0
                    for j in range(numSamples):
                        result = agent.evaluate()["env_runners"]
                        averageReturn += result["episode_return_mean"]
                        averageSteps += result["episode_len_mean"]
                    averageReturn = round(averageReturn / numSamples, 2)
                    print("Performance:", averageReturn)
                    averageSteps = round(averageSteps / numSamples, 0)
                    print("Steps:", averageSteps)
                    numSamples = (
                        int((evalMaxSteps - averageSteps) / evalMaxSteps * 10) + 1
                    )
                agent.save(checkpointPath)
