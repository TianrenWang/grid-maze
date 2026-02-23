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
        orthonormalLoss = 0
        placeCells = None
        for name, weight in parameters:
            if name == "moduleProjector.weight":
                P = torch.softmax(weight, dim=1)
                M = P.shape[0] // 2
                P_blocks = P.view(M, 2, -1)
                G = torch.einsum("mid,njd->mnij", P_blocks, P_blocks)
                G_sq = G.pow(2).sum(dim=(2, 3))
                orthogonalityLoss = G_sq.triu(diagonal=1).sum()
                orthonormalLoss += orthogonalityLoss / (M * (M - 1) / 2)

                identityMatrix = torch.eye(2, device=P.device)
                norm_loss = 0
                for i in range(M):
                    Pi = P[2 * i : 2 * i + 2]
                    norm_loss += (Pi @ Pi.T - identityMatrix).pow(2).sum()
                orthonormalLoss += norm_loss / M
                self.metrics.log_value(
                    key=(module_id, "orthonormal_loss"),
                    value=orthonormalLoss.cpu().detach().numpy(),
                    window=100,
                )
            if name == "placeCells":
                placeCells = weight

        lossMask = batch["loss_mask"]
        placeLogit = fwd_out["placeLogit"][lossMask]
        placeTarget = fwd_out["placeTarget"][lossMask]
        predictions = torch.nn.functional.softmax(placeLogit, -1)
        placeLoss = torch.nn.functional.cross_entropy(placeLogit, placeTarget)
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

        # Vision Loss
        obs: torch.Tensor = batch["obs"]
        module = self.module[module_id]
        visionSize = module.inputSize**2 * 3
        vision = (
            obs[:, :, visionSize:][lossMask]
            .reshape([-1, module.inputSize, module.inputSize, 3])[:, :, :, 0]
            .flatten(1)
        )
        predictedObs = fwd_out["predictedObs"][lossMask]
        visionReconstructionLoss = torch.nn.functional.binary_cross_entropy(
            predictedObs, vision
        )
        self.metrics.log_value(
            key=(module_id, "orthonormal_loss"),
            value=orthonormalLoss.cpu().detach().numpy(),
            window=100,
        )
        self.metrics.log_value(
            key=(module_id, "vision_reconstruction_loss"),
            value=visionReconstructionLoss.cpu().detach().numpy(),
            window=100,
        )

        if config.learner_config_dict.get("self_localize"):
            total_loss = placeLoss + orthonormalLoss + visionReconstructionLoss
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
