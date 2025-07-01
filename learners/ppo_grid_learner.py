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
        placeLogit = fwd_out["placeLogit"].flatten(0, 1)
        placeTarget = fwd_out["placeTarget"].flatten(0, 1)
        numCells = placeLogit[0].shape[0]
        maskTensor = torch.tensor([[1 / numCells] * numCells], device=placeLogit.device)
        maskIndices = batch["loss_mask"].to(torch.int16).flatten() == 0
        placeLogit[maskIndices] = maskTensor
        placeTarget[maskIndices] = maskTensor
        placeLoss = torch.nn.functional.cross_entropy(placeLogit, placeTarget)
        localization_coeff = config.learner_config_dict["localization_coeff"]
        self_localize = config.learner_config_dict.get("self_localize")
        if self_localize:
            total_loss = localization_coeff * placeLoss
        else:
            base_total_loss = super().compute_loss_for_module(
                module_id=module_id,
                config=config,
                batch=batch,
                fwd_out=fwd_out,
            )
            total_loss = base_total_loss
        self.metrics.log_value(
            key=(module_id, "localization_loss"),
            value=placeLoss,
            window=1,
        )
        targetCounts = torch.sum(placeTarget, dim=0) / torch.sum(placeTarget)
        placeBias = torch.max(targetCounts) - torch.min(targetCounts)
        self.metrics.log_value(
            key=(module_id, "place_bias"),
            value=placeBias,
            window=1,
        )
        return total_loss
