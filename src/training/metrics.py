from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
from torch import Tensor


class SegmentationMetricTracker:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)
        self.total_loss = 0.0
        self.total_main_loss = 0.0
        self.total_aux_loss = 0.0
        self.total_batches = 0
        self.total_examples = 0
        self.total_valid_pixels = 0
        self.total_correct_pixels = 0
        self.total_step_time_sec = 0.0
        self.total_forward_time_sec = 0.0
        self.total_backward_time_sec = 0.0
        self.total_optimizer_time_sec = 0.0

    def update_confusion_matrix(self, predictions: Tensor, targets: Tensor) -> None:
        predictions = predictions.detach().to(torch.int64).cpu().view(-1)
        targets = targets.detach().to(torch.int64).cpu().view(-1)
        valid = targets != self.ignore_index
        predictions = predictions[valid]
        targets = targets[valid]
        if targets.numel() == 0:
            return
        encoded = targets * self.num_classes + predictions
        bins = torch.bincount(encoded, minlength=self.num_classes * self.num_classes)
        self.confusion_matrix += bins.reshape(self.num_classes, self.num_classes).to(torch.float64)
        self.total_valid_pixels += int(targets.numel())
        self.total_correct_pixels += int((predictions == targets).sum().item())

    def update_from_step(self, step_metrics: Dict[str, Any]) -> None:
        self.total_batches += 1
        self.total_examples += int(step_metrics.get("batch_size", 0))
        self.total_loss += float(step_metrics.get("loss", 0.0))
        self.total_main_loss += float(step_metrics.get("main_loss", 0.0))
        aux_loss = step_metrics.get("aux_loss")
        if aux_loss is not None:
            self.total_aux_loss += float(aux_loss)
        self.total_step_time_sec += float(step_metrics.get("step_time_sec", 0.0))
        self.total_forward_time_sec += float(step_metrics.get("forward_time_sec", 0.0))
        self.total_backward_time_sec += float(step_metrics.get("backward_time_sec", 0.0))
        self.total_optimizer_time_sec += float(step_metrics.get("optimizer_time_sec", 0.0))

    def compute(self, class_names: Iterable[str] | None = None) -> Dict[str, Any]:
        diag = torch.diag(self.confusion_matrix)
        gt = self.confusion_matrix.sum(dim=1)
        pred = self.confusion_matrix.sum(dim=0)
        union = gt + pred - diag

        per_class_iou = torch.where(union > 0, diag / union.clamp_min(1.0), torch.full_like(union, float("nan")))
        mean_iou = torch.nanmean(per_class_iou).item() if torch.any(~torch.isnan(per_class_iou)) else 0.0
        pixel_accuracy = (
            float(self.total_correct_pixels / self.total_valid_pixels)
            if self.total_valid_pixels > 0
            else 0.0
        )
        mean_loss = self.total_loss / max(self.total_batches, 1)
        mean_main_loss = self.total_main_loss / max(self.total_batches, 1)
        mean_aux_loss = self.total_aux_loss / max(self.total_batches, 1)

        metrics: Dict[str, Any] = {
            "mean_loss": mean_loss,
            "mean_main_loss": mean_main_loss,
            "mean_aux_loss": mean_aux_loss,
            "pixel_accuracy": pixel_accuracy,
            "miou": mean_iou,
            "num_batches": self.total_batches,
            "num_examples": self.total_examples,
            "valid_pixels": self.total_valid_pixels,
            "correct_pixels": self.total_correct_pixels,
            "epoch_time_sec": self.total_step_time_sec,
            "mean_step_time_sec": self.total_step_time_sec / max(self.total_batches, 1),
            "mean_forward_time_sec": self.total_forward_time_sec / max(self.total_batches, 1),
            "mean_backward_time_sec": self.total_backward_time_sec / max(self.total_batches, 1),
            "mean_optimizer_time_sec": self.total_optimizer_time_sec / max(self.total_batches, 1),
            "examples_per_sec": float(self.total_examples / max(self.total_step_time_sec, 1e-12)),
            "per_class_iou": per_class_iou.tolist(),
        }

        if class_names is not None:
            names: List[str] = list(class_names)
            metrics["per_class_iou_named"] = {
                names[idx] if idx < len(names) else f"class_{idx}": value
                for idx, value in enumerate(per_class_iou.tolist())
            }

        return metrics
