from typing import Any, Dict

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID

import torch


class PPOTorchLearnerWithSelfPredLoss(PPOTorchLearner):
    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, torch.Tensor],
    ):
        base_total_loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        placeLogit = fwd_out["placeLogit"].flatten(0, 1)
        placeTarget = fwd_out["placeTarget"].flatten(0, 1)
        numCells = placeLogit[0].shape[0]
        maskTensor = torch.Tensor([[1 / numCells] * numCells])
        maskIndices = batch["loss_mask"].to(torch.int16).flatten() == 0
        placeLogit[maskIndices] = maskTensor
        placeTarget[maskIndices] = maskTensor
        placeLoss = torch.nn.functional.cross_entropy(placeLogit, placeTarget)
        localization_coeff = config.learner_config_dict["localization_coeff"]
        total_loss = base_total_loss + localization_coeff * placeLoss

        return total_loss
