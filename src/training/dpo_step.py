from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import GradScaler

from .step import (
    StepConfig,
    _build_memory_stats,
    _sync_if_cuda,
    compute_grad_norm,
    estimate_optimizer_state_bytes,
)


@dataclass
class DPOStepConfig:
    beta: float = 0.1
    zero_grad_set_to_none: bool = True
    track_grad_norm: bool = True
    grad_norm_type: float = 2.0
    use_amp: bool = False


def _get_sequence_logprobs(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
) -> Tensor:
    """
    Returns per-sequence sum of log-probs over response tokens (label != -100).
    Shape: (B,)

    Shifts logits left by one so position t predicts token t+1. Prompt positions
    (label == -100) are masked out and contribute zero to the sum.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)

    shift_logits = logits[:, :-1, :]   # (B, L-1, V)
    shift_labels = labels[:, 1:]        # (B, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Replace -100 with 0 before gather to avoid index error; masked out below.
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0
    token_logprobs = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

    response_mask = (shift_labels != -100).float()
    return (token_logprobs * response_mask).sum(dim=-1)  # (B,)


def _compute_dpo_loss(
    chosen_logprobs: Tensor,
    rejected_logprobs: Tensor,
    ref_chosen_logprobs: Tensor,
    ref_rejected_logprobs: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    DPO objective: -E[log σ(β((log π_θ(yw|x) - log π_ref(yw|x))
                               - (log π_θ(yl|x) - log π_ref(yl|x))))]
    Returns (loss, chosen_rewards, rejected_rewards).
    """
    chosen_rewards = beta * (chosen_logprobs - ref_chosen_logprobs)
    rejected_rewards = beta * (rejected_logprobs - ref_rejected_logprobs)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, chosen_rewards, rejected_rewards


def dpo_train_step(
    policy: nn.Module,
    ref_model: nn.Module,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    config: DPOStepConfig | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = DPOStepConfig()

    policy.train()
    ref_model.eval()
    optimizer.zero_grad(set_to_none=config.zero_grad_set_to_none)

    chosen_input_ids = batch["chosen_input_ids"].to(device)
    chosen_attention_mask = batch["chosen_attention_mask"].to(device)
    chosen_labels = batch["chosen_labels"].to(device)
    rejected_input_ids = batch["rejected_input_ids"].to(device)
    rejected_attention_mask = batch["rejected_attention_mask"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)

    batch_size = chosen_input_ids.shape[0]
    num_tokens = int(
        (chosen_labels != -100).sum().item() + (rejected_labels != -100).sum().item()
    )

    # Concatenate chosen + rejected along the batch dimension for a single forward pass.
    combined_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
    combined_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
    combined_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync_if_cuda(device)
    step_start = time.perf_counter()

    amp_enabled = bool(config.use_amp and device.type == "cuda")
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        all_logprobs = _get_sequence_logprobs(policy, combined_input_ids, combined_attention_mask, combined_labels)
        chosen_logprobs, rejected_logprobs = all_logprobs.chunk(2, dim=0)

    with torch.no_grad():
        ref_all_logprobs = _get_sequence_logprobs(ref_model, combined_input_ids, combined_attention_mask, combined_labels)
        ref_chosen_logprobs, ref_rejected_logprobs = ref_all_logprobs.chunk(2, dim=0)

    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        loss, chosen_rewards, rejected_rewards = _compute_dpo_loss(
            chosen_logprobs, rejected_logprobs,
            ref_chosen_logprobs, ref_rejected_logprobs,
            beta=config.beta,
        )

    _sync_if_cuda(device)
    forward_end = time.perf_counter()

    if scaler is not None and amp_enabled:
        scaler.scale(loss).backward()
        if config.track_grad_norm:
            scaler.unscale_(optimizer)
        _sync_if_cuda(device)
        backward_end = time.perf_counter()
        grad_norm = compute_grad_norm(policy, norm_type=config.grad_norm_type) if config.track_grad_norm else None
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        _sync_if_cuda(device)
        backward_end = time.perf_counter()
        grad_norm = compute_grad_norm(policy, norm_type=config.grad_norm_type) if config.track_grad_norm else None
        optimizer.step()

    _sync_if_cuda(device)
    step_end = time.perf_counter()

    chosen_rewards_d = chosen_rewards.detach()
    rejected_rewards_d = rejected_rewards.detach()

    metrics: Dict[str, Any] = {
        "dpo_loss": float(loss.detach().item()),
        "chosen_logprob": float(chosen_logprobs.detach().mean().item()),
        "rejected_logprob": float(rejected_logprobs.detach().mean().item()),
        "chosen_reward": float(chosen_rewards_d.mean().item()),
        "rejected_reward": float(rejected_rewards_d.mean().item()),
        "preference_correct": int((chosen_rewards_d > rejected_rewards_d).sum().item()),
        "batch_size": batch_size,
        "num_tokens": num_tokens,
        "forward_time_sec": forward_end - step_start,
        "backward_time_sec": backward_end - forward_end,
        "optimizer_time_sec": step_end - backward_end,
        "step_time_sec": step_end - step_start,
        "grad_norm": grad_norm,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "optimizer_state_bytes": estimate_optimizer_state_bytes(optimizer),
    }
    metrics.update(_build_memory_stats(device))
    return metrics
