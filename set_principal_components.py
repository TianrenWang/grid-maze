from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID

import numpy as np
import torch
import argparse
import os

from environments import FoggedMazeEnv
import models

parser = argparse.ArgumentParser()
parser.add_argument("--expName", type=str, default="default_exp")
args = parser.parse_args()

NUM_MODULES = 5


def setPrincipalComponents(mazeSize: int, modulePath: str, expName: str):
    env = FoggedMazeEnv(
        {
            "maze": None,
            "goal": (mazeSize // 2, mazeSize // 2),
            "start": None,
            "maxSteps": 1000,
            "mazeSize": mazeSize,
        }
    )
    module: models.LatentPathModule = RLModule.from_checkpoint(modulePath)
    obs, _ = env.reset()
    episodes = 0
    latents = []

    while episodes < 2000:
        if episodes % 100 == 0:
            print(episodes)
        previousState = module.get_initial_state()
        obs, _ = env.reset()
        done = False
        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
            batched_obs = {
                Columns.OBS: obs,
                Columns.STATE_IN: {
                    k: torch.reshape(v, [1, -1]) for k, v in previousState.items()
                },
            }
            rl_module_out = module.forward(batched_obs)
            latents.append(rl_module_out["latents"].squeeze(0))
            action = np.random.choice(
                4,
                p=torch.softmax(
                    rl_module_out[Columns.ACTION_DIST_INPUTS].flatten(), dim=0
                )
                .detach()
                .cpu()
                .numpy(),
            )
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated
            previousState = rl_module_out[Columns.STATE_OUT]
        episodes += 1

    samples = torch.concat(latents, dim=0)
    module.moduleProjector.update(samples)
    module.save_to_path(modulePath)


if __name__ == "__main__":
    mazeSize = 30
    checkpointPath = f"{os.path.abspath(os.getcwd())}/checkpoints/{args.expName}"
    rlModulePath = os.path.join(
        checkpointPath,
        "learner_group",
        "learner",
        "rl_module",
        DEFAULT_MODULE_ID,
    )
    setPrincipalComponents(
        mazeSize,
        rlModulePath,
        args.expName,
    )
