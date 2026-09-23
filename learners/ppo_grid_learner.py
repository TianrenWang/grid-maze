from typing import Any

import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID

import models


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
        module = self.module[module_id]
        lossMask = batch["loss_mask"]
        if config.learner_config_dict.get("self_localize"):
            loss = 0
            if "placeLogit" in fwd_out and "placeTarget" in fwd_out:
                parameters = module.named_parameters()
                placeCells = None
                for name, weight in parameters:
                    if name == "placeCells":
                        placeCells = weight

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
        elif "jepaLoss" in fwd_out:
            jepaLoss: torch.Tensor = fwd_out["jepaLoss"][lossMask].mean()
            self.metrics.log_value(
                key=(module_id, "jepa_loss"),
                value=jepaLoss.cpu().detach().numpy(),
                window=100,
            )
            _, _, coordinates, _ = self.module[module_id]._getObsFromBatch(batch)
            coordinates = coordinates[lossMask]
            predictedCoordinates: torch.Tensor = fwd_out["coordinateReadout"][lossMask]
            coordinateLoss = torch.abs(coordinates - predictedCoordinates).mean()
            self.metrics.log_value(
                key=(module_id, "coordinate_loss"),
                value=coordinateLoss.cpu().detach().numpy(),
                window=100,
            )
            return jepaLoss + coordinateLoss
        elif "manifoldCoordinate" in fwd_out:
            jepaLatents = fwd_out["jepaLatent"][lossMask]
            manifoldCoordinates = fwd_out["manifoldCoordinate"][lossMask]
            normalizedLatents = torch.nn.functional.normalize(jepaLatents, dim=-1)
            similarity = normalizedLatents @ normalizedLatents.T
            distances = torch.cdist(manifoldCoordinates, manifoldCoordinates)
            deduplicationMask = torch.triu(
                torch.ones_like(similarity, dtype=torch.bool), diagonal=1
            )
            similarity = similarity[deduplicationMask]
            distances = distances[deduplicationMask]
            similarityPrediction = module.similarityPredictor(distances.reshape(-1, 1))
            targetSimilarity = (similarity > 0.95).float()
            negativeSampleLoss = torch.nn.functional.binary_cross_entropy(
                similarityPrediction.flatten(), targetSimilarity
            )

            movements = torch.diff(fwd_out["manifoldCoordinate"], dim=1)[
                lossMask[:, 1:]
            ]
            distances = torch.linalg.norm(movements, dim=-1)
            distanceLoss = torch.mean(((distances - module.speed) / module.speed) ** 2)

            self.metrics.log_value(
                key=(module_id, "negative_sample_loss"),
                value=negativeSampleLoss.cpu().detach().numpy(),
                window=100,
            )
            self.metrics.log_value(
                key=(module_id, "distance_loss"),
                value=distanceLoss.cpu().detach().numpy(),
                window=100,
            )
            return negativeSampleLoss + distanceLoss
        else:
            return super().compute_loss_for_module(
                module_id=module_id,
                config=config,
                batch=batch,
                fwd_out=fwd_out,
            )

    @override(PPOTorchLearner)
    def apply_gradients(self, gradients_dict: dict[str, Any]) -> None:
        super().apply_gradients(gradients_dict)
        module = self.module[DEFAULT_MODULE_ID]
        if (
            type(module) is models.LatentPathModule
            and module.trainingPhase == "learnJEPA"
        ):
            module.jepa.EMAEncoder.update()
