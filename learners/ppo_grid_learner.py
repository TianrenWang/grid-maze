import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID


class PPOTorchLearnerWithSelfPredLoss(PPOTorchLearner):
    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: dict[str, dict],
        fwd_out: dict[str, torch.Tensor],
    ):
        lossMask = batch["loss_mask"]
        parameters = self.module[module_id].named_parameters()
        placeCells = None
        for name, weight in parameters:
            if name == "placeCells":
                placeCells = weight

        if config.learner_config_dict.get("self_localize"):
            loss = 0

            if "placeLogit" in fwd_out and "placeTarget" in fwd_out:
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
                loss += placeLoss
                positionError = torch.mean(
                    torch.sqrt(
                        torch.sum(
                            (decodedPredictedPositions - decodedActualPositions) ** 2,
                            -1,
                        )
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
                self.metrics.log_value(
                    key=(module_id, "prediction_error"),
                    value=predictionError.cpu().detach().numpy(),
                    window=100,
                )

            return loss
        else:
            loss = super().compute_loss_for_module(
                module_id=module_id,
                config=config,
                batch=batch,
                fwd_out=fwd_out,
            )

            if "trueCoordinates" in fwd_out and "manifoldCoordinates" in fwd_out:
                trueCoordinates = fwd_out["trueCoordinates"][lossMask]
                manifoldCoordinates = fwd_out["manifoldCoordinates"][lossMask]
                relativeDifferences = torch.linalg.norm(
                    trueCoordinates - manifoldCoordinates, dim=-1
                )
                coherenceLoss = relativeDifferences.mean()
                loss += coherenceLoss
                self.metrics.log_value(
                    key=(module_id, "coherence_loss"),
                    value=coherenceLoss.cpu().detach().numpy(),
                    window=100,
                )

            if "movements" in fwd_out:
                movements: torch.Tensor = fwd_out["movements"][lossMask]
                distances = torch.linalg.norm(movements, dim=1)
                # idealDistance = torch.mean(distances).detach()
                idealDistance = 1 / 30  # switch to the other loss when I am ready
                movement_loss = torch.mean(
                    ((distances - idealDistance) / idealDistance) ** 2
                )
                loss += movement_loss
                self.metrics.log_value(
                    key=(module_id, "movement_loss"),
                    value=movement_loss.cpu().detach().numpy(),
                    window=100,
                )

            if "rewards" in fwd_out:
                nonRewardMask = fwd_out["rewards"] != 0
                manifold = fwd_out["manifold"]
                rewardedCoordinates = manifold[nonRewardMask]
                if rewardedCoordinates.numel() != 0:
                    # likelyLandmarkCoordinate = torch.mean(
                    #     rewardedCoordinates, dim=0
                    # ).detach()
                    coherenceLoss = torch.nn.functional.l1_loss(
                        rewardedCoordinates,
                        torch.tensor(
                            [[0.5, 0.5]],
                            dtype=torch.float32,
                            device=rewardedCoordinates.device,
                        ).expand_as(rewardedCoordinates),
                    )
                    loss += coherenceLoss
                    self.metrics.log_value(
                        key=(module_id, "coherence_loss"),
                        value=coherenceLoss.cpu().detach().numpy(),
                        window=100,
                    )

            return loss
