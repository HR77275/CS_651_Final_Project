from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
from torch import Tensor, nn
from torch.amp import GradScaler


@dataclass
class StepConfig:
    ignore_index: int = 255
    aux_weight: float = 0.4
    zero_grad_set_to_none: bool = True
    track_grad_norm: bool = True
    grad_norm_type: float = 2.0
    use_amp: bool = False
    return_predictions: bool = False


def build_segmentation_loss(ignore_index: int = 255) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=ignore_index)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def estimate_parameter_bytes(model: nn.Module) -> int:
    return sum(_tensor_bytes(param) for param in model.parameters())


def estimate_gradient_bytes(model: nn.Module) -> int:
    total = 0
    for param in model.parameters():
        if param.grad is not None:
            total += _tensor_bytes(param.grad)
    return total


def estimate_optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += _tensor_bytes(value)
    return total


def compute_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    grads = [param.grad.detach() for param in model.parameters() if param.grad is not None]
    if not grads:
        return 0.0
    if math.isinf(norm_type):
        return max(grad.abs().max().item() for grad in grads)
    device = grads[0].device
    norms = torch.stack([torch.norm(grad, p=norm_type).to(device) for grad in grads])
    return torch.norm(norms, p=norm_type).item()


def compute_parameter_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    params = [param.detach() for param in model.parameters()]
    if not params:
        return 0.0
    if math.isinf(norm_type):
        return max(param.abs().max().item() for param in params)
    device = params[0].device
    norms = torch.stack([torch.norm(param, p=norm_type).to(device) for param in params])
    return torch.norm(norms, p=norm_type).item()


def _compute_losses(
    outputs: Mapping[str, Tensor],
    targets: Tensor,
    criterion: nn.Module,
    aux_weight: float,
) -> tuple[Tensor, Tensor, Tensor | None]:
    main_loss = criterion(outputs["out"], targets)
    aux_loss = None
    if "aux" in outputs and outputs["aux"] is not None:
        aux_loss = criterion(outputs["aux"], targets)
        total_loss = main_loss + aux_weight * aux_loss
    else:
        total_loss = main_loss
    return total_loss, main_loss, aux_loss


def _build_memory_stats(device: torch.device) -> Dict[str, int | None]:
    if device.type != "cuda":
        return {
            "gpu_memory_allocated_bytes": None,
            "gpu_memory_reserved_bytes": None,
            "gpu_peak_allocated_bytes": None,
            "gpu_peak_reserved_bytes": None,
        }
    return {
        "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(device),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def train_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module | None = None,
    scaler: GradScaler | None = None,
    config: StepConfig | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = StepConfig()
    if criterion is None:
        criterion = build_segmentation_loss(ignore_index=config.ignore_index)

    model.train()
    optimizer.zero_grad(set_to_none=config.zero_grad_set_to_none)

    batch = _move_batch_to_device(batch, device)
    images = batch["image"]
    targets = batch["mask"].long()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync_if_cuda(device)

    step_start = time.perf_counter()
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        outputs = model(images)
        total_loss, main_loss, aux_loss = _compute_losses(
            outputs=outputs,
            targets=targets,
            criterion=criterion,
            aux_weight=config.aux_weight,
        )
    _sync_if_cuda(device)
    forward_end = time.perf_counter()

    if scaler is not None and amp_enabled:
        scaler.scale(total_loss).backward()
        if config.track_grad_norm:
            scaler.unscale_(optimizer)
        _sync_if_cuda(device)
        backward_end = time.perf_counter()
        grad_norm = compute_grad_norm(model, norm_type=config.grad_norm_type) if config.track_grad_norm else None
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        _sync_if_cuda(device)
        backward_end = time.perf_counter()
        grad_norm = compute_grad_norm(model, norm_type=config.grad_norm_type) if config.track_grad_norm else None
        optimizer.step()

    _sync_if_cuda(device)
    step_end = time.perf_counter()

    predictions = outputs["out"].argmax(dim=1)
    valid_mask = targets != config.ignore_index
    valid_pixels = int(valid_mask.sum().item())
    correct_pixels = int(((predictions == targets) & valid_mask).sum().item())

    metrics: Dict[str, Any] = {
        "loss": float(total_loss.detach().item()),
        "main_loss": float(main_loss.detach().item()),
        "aux_loss": None if aux_loss is None else float(aux_loss.detach().item()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "batch_size": int(images.shape[0]),
        "image_size": tuple(images.shape[-2:]),
        "forward_time_sec": forward_end - step_start,
        "backward_time_sec": backward_end - forward_end,
        "optimizer_time_sec": step_end - backward_end,
        "step_time_sec": step_end - step_start,
        "examples_per_sec": float(images.shape[0] / max(step_end - step_start, 1e-12)),
        "valid_pixels": valid_pixels,
        "correct_pixels": correct_pixels,
        "pixel_accuracy": float(correct_pixels / valid_pixels) if valid_pixels > 0 else 0.0,
        "grad_norm": grad_norm,
        "parameter_norm": compute_parameter_norm(model, norm_type=config.grad_norm_type),
        "parameter_bytes": estimate_parameter_bytes(model),
        "gradient_bytes": estimate_gradient_bytes(model),
        "optimizer_state_bytes": estimate_optimizer_state_bytes(optimizer),
        "input_bytes": _tensor_bytes(images) + _tensor_bytes(targets),
    }
    metrics.update(_build_memory_stats(device))

    if config.return_predictions:
        metrics["predictions"] = predictions.detach().cpu()
        metrics["targets"] = targets.detach().cpu()

    return metrics

