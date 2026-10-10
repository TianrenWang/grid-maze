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
        lossMask: torch.Tensor = batch["loss_mask"]
        _, _, coordinates, _ = self.module[module_id]._getObsFromBatch(batch)
        coordinates = coordinates[lossMask]

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
            predictedCoordinates: torch.Tensor = fwd_out["coordinateReadout"][lossMask]
            coordinateLoss = torch.abs(coordinates - predictedCoordinates).mean()
            self.metrics.log_value(
                key=(module_id, "coordinate_loss"),
                value=coordinateLoss.cpu().detach().numpy(),
                window=100,
            )
            return jepaLoss + coordinateLoss
        elif module.trainingPhase == "learnMovement":
            matrixLossMask = lossMask.float().unsqueeze(-1)
            matrixLossMask = (matrixLossMask @ matrixLossMask.transpose(-1, -2)).bool()
            jepaLatents = torch.nn.functional.normalize(fwd_out["jepaLatent"], dim=-1)
            similarity = jepaLatents @ jepaLatents.transpose(-1, -2)
            calculatedCoordinates = fwd_out["calculatedManifold"]
            distances = torch.cdist(calculatedCoordinates, calculatedCoordinates)
            deduplicationMask = torch.triu(
                torch.ones_like(similarity, dtype=torch.bool), diagonal=1
            )
            matrixLossMask = matrixLossMask[deduplicationMask]
            similarity = similarity[deduplicationMask][matrixLossMask]
            distances = distances[deduplicationMask][matrixLossMask]
            samenessThreshold = 0.995
            sameObsMask = similarity >= samenessThreshold
            diffObsMask = (similarity < samenessThreshold) & (
                distances < module.manifoldProjector.speed * 1.415
            )
            distancesOfSameObs = distances[sameObsMask].mean()
            distancesOfDiffObs = distances[diffObsMask][:150]
            negativeSampleLoss = torch.exp(-distancesOfDiffObs * 50)

            pastSameObsLoss = self.metrics.peek((module_id, "sameObsCoherenceLoss"), 1)
            reachedStableState = self.metrics.peek((module_id, "stable"), 0)

            if pastSameObsLoss <= 1e-3 and not reachedStableState:
                print("Reached stable movement state")
                self.metrics.log_value(key=(module_id, "stable"), value=1, window=1)
                reachedStableState = 1

            directionScore = module.manifoldProjector._getCrossScore()
            self.metrics.log_value(
                key=(module_id, "sameObsCoherenceLoss"),
                value=distancesOfSameObs.cpu().detach().numpy(),
                window=100,
            )
            self.metrics.log_value(
                key=(module_id, "diffObsCoherenceLoss"),
                value=negativeSampleLoss.sum().cpu().detach().numpy(),
                window=100,
            )
            self.metrics.log_value(
                key=(module_id, "directionScore"),
                value=directionScore.cpu().detach().numpy(),
                window=100,
            )

            if reachedStableState:
                return distancesOfSameObs + negativeSampleLoss.mean()
            else:
                return distancesOfSameObs
        elif module.trainingPhase == "learnManifold":
            # Projection accuracy
            calculatedCoordinates = fwd_out["calculatedManifold"][lossMask]
            projectedManifold = fwd_out["projectedManifold"][lossMask]
            projectionError = torch.abs(
                projectedManifold - calculatedCoordinates.detach()
            )
            self.metrics.log_value(
                key=(module_id, "projectionError"),
                value=projectionError.mean().cpu().detach().numpy(),
                window=100,
            )

            # Global consistency
            calculatedCoordinates = fwd_out["calculatedManifold"][lossMask]
            predictedDistances = torch.cdist(
                calculatedCoordinates, calculatedCoordinates
            )
            trueDistances = torch.cdist(coordinates, coordinates)
            deduplicationMask = torch.triu(
                torch.ones_like(trueDistances, dtype=torch.bool), diagonal=1
            )
            predictedDistances = predictedDistances[deduplicationMask]
            trueDistances = trueDistances[deduplicationMask]
            inconsistency = torch.abs(trueDistances - predictedDistances).mean()
            self.metrics.log_value(
                key=(module_id, "inconsistency"),
                value=inconsistency.mean().cpu().detach().numpy(),
                window=100,
            )
            return projectionError.mean()
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
        if type(module) is models.LatentPathModule:
            if module.trainingPhase == "learnJEPA":
                module.jepa.EMAEncoder.update()
            elif module.trainingPhase == "learnManifold":
                module.manifoldProjector.EMAProjector.update()
