from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from .metrics import SegmentationMetricTracker
from .step import StepConfig, build_segmentation_loss, train_step


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
    class_names: Iterable[str] | None = None,
    config: StepConfig | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = StepConfig()
    if criterion is None:
        criterion = build_segmentation_loss(ignore_index=config.ignore_index)

    tracker = SegmentationMetricTracker(num_classes=num_classes, ignore_index=config.ignore_index)
    model.eval()

    for batch in dataloader:
        moved_batch: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved_batch[key] = value.to(device, non_blocking=device.type == "cuda")
            else:
                moved_batch[key] = value

        images = moved_batch["image"]
        targets = moved_batch["mask"].long()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=bool(config.use_amp and device.type == "cuda")):
            outputs = model(images)
            main_loss = criterion(outputs["out"], targets)
            aux_loss = criterion(outputs["aux"], targets) if "aux" in outputs and outputs["aux"] is not None else None
            total_loss = main_loss + config.aux_weight * aux_loss if aux_loss is not None else main_loss
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        predictions = outputs["out"].argmax(dim=1)
        tracker.update_confusion_matrix(predictions=predictions, targets=targets)
        tracker.update_from_step(
            {
                "loss": float(total_loss.item()),
                "main_loss": float(main_loss.item()),
                "aux_loss": None if aux_loss is None else float(aux_loss.item()),
                "batch_size": int(images.shape[0]),
                "step_time_sec": elapsed,
                "forward_time_sec": elapsed,
                "backward_time_sec": 0.0,
                "optimizer_time_sec": 0.0,
            }
        )

    metrics = tracker.compute(class_names=class_names)
    metrics["split"] = "val"
    return metrics


def summarize_training_epoch(
    step_metrics_list: Iterable[Mapping[str, Any]],
    class_names: Iterable[str] | None = None,
    num_classes: int = 21,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    tracker = SegmentationMetricTracker(num_classes=num_classes, ignore_index=ignore_index)
    for step_metrics in step_metrics_list:
        tracker.update_from_step(dict(step_metrics))
        predictions = step_metrics.get("predictions")
        targets = step_metrics.get("targets")
        if predictions is not None and targets is not None:
            tracker.update_confusion_matrix(predictions=predictions, targets=targets)

    metrics = tracker.compute(class_names=class_names)
    metrics["split"] = "train"
    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
    scaler: GradScaler | None = None,
    class_names: Iterable[str] | None = None,
    config: StepConfig | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = StepConfig()
    if criterion is None:
        criterion = build_segmentation_loss(ignore_index=config.ignore_index)

    tracker = SegmentationMetricTracker(num_classes=num_classes, ignore_index=config.ignore_index)
    epoch_config = StepConfig(
        ignore_index=config.ignore_index,
        aux_weight=config.aux_weight,
        zero_grad_set_to_none=config.zero_grad_set_to_none,
        track_grad_norm=config.track_grad_norm,
        grad_norm_type=config.grad_norm_type,
        use_amp=config.use_amp,
        return_predictions=True,
    )
    last_step_metrics: Dict[str, Any] | None = None

    for batch in dataloader:
        step_metrics = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            scaler=scaler,
            config=epoch_config,
        )
        tracker.update_from_step(step_metrics)
        tracker.update_confusion_matrix(
            predictions=step_metrics["predictions"],
            targets=step_metrics["targets"],
        )
        last_step_metrics = step_metrics

    metrics = tracker.compute(class_names=class_names)
    metrics["split"] = "train"
    metrics["optimizer_state_bytes"] = last_step_metrics.get("optimizer_state_bytes") if last_step_metrics is not None else 0
    metrics["parameter_bytes"] = last_step_metrics.get("parameter_bytes") if last_step_metrics is not None else 0
    return metrics

