import argparse
import os

import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import models
from environments import PlaceMazeEnv

parser = argparse.ArgumentParser()
parser.add_argument("--mazeSize", type=int, default=30)
parser.add_argument("--hiddenSize", type=int, default=32)
parser.add_argument("--numLayers", type=int, default=2)
parser.add_argument("--maxSteps", type=int, default=1000)
parser.add_argument("--expName", type=str, default="default_exp")
parser.add_argument("--grid", action="store_true")
parser.add_argument("--memoryLen", type=int, default=20)
parser.add_argument("--latentPath", action="store_true")
parser.add_argument("--pretraining", action="store_true")
parser.add_argument("--perturb", action="store_true")
parser.add_argument("--visionPolicy", action="store_true")
parser.add_argument("--integrationPolicy", action="store_true")
parser.add_argument("--debug", type=int, default=0)
args = parser.parse_args()

if args.pretraining:
    args.latentPath = True


def usesGrid():
    return args.grid or args.latentPath or args.integrationPolicy


if __name__ == "__main__":
    mazeSize = args.mazeSize
    visionRange = 4

    if usesGrid():
        module = models.PathIntegrationWithVisionModuleForEval
        if args.latentPath:
            module = models.LatentPathModule
    else:
        module = models.PlaceCellControlModule

    env = PlaceMazeEnv

    environmentConfig = {
        "maze": None,
        "start": None,
        "maxSteps": 300,
        "memoryLen": args.memoryLen,
        "mazeSize": mazeSize,
        "perturb": args.perturb,
        "eval": True,
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
                    "inputSize": visionRange * 2 + 1,
                    "max_seq_len": args.memoryLen,
                    "mazeSize": mazeSize,
                    "visionPolicy": args.visionPolicy,
                    "integrationPolicy": args.integrationPolicy,
                },
            ),
        )
        .learners(num_gpus_per_learner=1 if torch.cuda.is_available() else 0)
        .evaluation(
            evaluation_num_env_runners=1 if args.debug else 8,
            evaluation_duration_unit="episodes",
            evaluation_duration=1 if args.debug else 128,
        )
    )
    agentConfig.env_config = environmentConfig
    agent = agentConfig.build_algo()
    checkpointPath = f"{os.path.abspath(os.getcwd())}/checkpoints/{args.expName}"
    if os.path.exists(checkpointPath):
        agent.restore_from_path(checkpointPath)

    if args.debug:
        for i in range(args.debug):
            agent.evaluate()
    else:
        numSamples = 10
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
        numSamples = int((args.maxSteps - averageSteps) / args.maxSteps * 10) + 1
