import argparse
import os
import pickle

import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec

import models
from environments import MazeEnv

parser = argparse.ArgumentParser()
parser.add_argument("--expName", type=str, default="default_exp")
parser.add_argument("--pretrainedName", type=str, default="default_exp")
args = parser.parse_args()


if __name__ == "__main__":
    folderName = f"{os.path.abspath(os.getcwd())}/checkpoints/"
    targetPath = f"{folderName}{args.expName}"
    sourcePath = f"{folderName}{args.pretrainedName}"
    targetModulePath = os.path.join(
        targetPath,
        "learner_group",
        "learner",
        "rl_module",
        DEFAULT_MODULE_ID,
    )
    sourceModulePath = os.path.join(
        sourcePath,
        "learner_group",
        "learner",
        "rl_module",
        DEFAULT_MODULE_ID,
    )

    if not os.path.exists(targetPath):
        agentConfig = (
            PPOConfig()
            .environment(MazeEnv)
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=models.LatentPathModule,
                    model_config={"hiddenSize": 32, "inputSize": 4 * 2 + 1},
                ),
            )
        )
        agent = agentConfig.build_algo()
        agent.save(targetPath)

    targetModule: models.LatentPathModule = RLModule.from_checkpoint(targetModulePath)

    with open(os.path.join(sourceModulePath, "module_state.pkl"), "rb") as f:
        sourceState = pickle.load(f)

    emaState = {
        k.removeprefix("jepa.EMAEncoder.ema_model."): torch.as_tensor(v)
        for k, v in sourceState.items()
        if k.startswith("jepa.EMAEncoder.ema_model.")
    }
    targetModule.jepa.EMAEncoder.ema_model.load_state_dict(emaState)
    targetModule.save_to_path(targetModulePath)
