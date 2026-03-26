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
        parameters = self.module[module_id].named_parameters()
        placeCells = None
        for name, weight in parameters:
            if name == "placeCells":
                placeCells = weight

        lossMask = batch["loss_mask"]
        placeLogit = fwd_out["placeLogit"][lossMask]
        placeTarget = fwd_out["placeTarget"][lossMask]
        predictions = torch.nn.functional.softmax(placeLogit, -1)
        placeLoss = torch.nn.functional.cross_entropy(
            placeLogit.flatten(0, 1), placeTarget.flatten(0, 1)
        )
        if len(placeCells.shape) == 2:
            decodedPredictedPositions = torch.matmul(predictions, placeCells)
            decodedActualPositions = torch.matmul(placeTarget, placeCells)
        else:
            decodedPredictedPositions = torch.einsum(
                "bmp,mpd->bmd", predictions, placeCells
            )
            decodedActualPositions = torch.einsum(
                "bmp,mpd->bmd", placeTarget, placeCells
            )
        positionError = torch.mean(
            torch.sqrt(
                torch.sum((decodedPredictedPositions - decodedActualPositions) ** 2, -1)
            )
        )
        self.metrics.log_value(
            key=(module_id, "position_error"),
            value=positionError.cpu().detach().numpy(),
            window=100,
        )
        predictionError = torch.mean(
            torch.sum(torch.abs(predictions - placeTarget), -1)
        )
        total_loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        if config.learner_config_dict.get("self_localize"):
            total_loss = placeLoss
        self.metrics.log_value(
            key=(module_id, "prediction_error"),
            value=predictionError.cpu().detach().numpy(),
            window=100,
        )
        # targetCounts = torch.sum(placeTarget, dim=0) / torch.sum(placeTarget)
        # placeBias = torch.max(targetCounts) - torch.min(targetCounts)
        # self.metrics.log_value(
        #     key=(module_id, "place_bias"),
        #     value=placeBias.cpu().detach().numpy(),
        #     window=100,
        # )
        return total_loss
