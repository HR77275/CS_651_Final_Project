from __future__ import annotations

import time
from typing import Any, Dict

import torch
from torch import nn
from torch.amp import GradScaler

from .dpo_step import DPOStepConfig, _compute_dpo_loss, _get_sequence_logprobs, dpo_train_step
from .step import _sync_if_cuda


def train_dpo_epoch(
    policy: nn.Module,
    ref_model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    config: DPOStepConfig | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = DPOStepConfig()

    # Deferred import: Person B owns dpo_metrics.py; mock DPOMetricTracker if not yet available.
    from .dpo_metrics import DPOMetricTracker

    tracker = DPOMetricTracker()
    policy.train()
    ref_model.eval()
    last_step_metrics: Dict[str, Any] | None = None

    for batch in dataloader:
        step_metrics = dpo_train_step(
            policy=policy,
            ref_model=ref_model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            config=config,
        )
        tracker.update(step_metrics)
        last_step_metrics = step_metrics

    metrics = tracker.compute()
    metrics["split"] = "train"
    metrics["optimizer_state_bytes"] = (
        last_step_metrics.get("optimizer_state_bytes", 0) if last_step_metrics else 0
    )
    return metrics


@torch.no_grad()
def evaluate_dpo_epoch(
    policy: nn.Module,
    ref_model: nn.Module,
    dataloader,
    device: torch.device,
    config: DPOStepConfig | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = DPOStepConfig()

    from .dpo_metrics import DPOMetricTracker

    tracker = DPOMetricTracker()
    policy.eval()
    ref_model.eval()

    for batch in dataloader:
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

        combined_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        combined_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        combined_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        _sync_if_cuda(device)
        start = time.perf_counter()

        amp_enabled = bool(config.use_amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            all_logprobs = _get_sequence_logprobs(policy, combined_input_ids, combined_attention_mask, combined_labels)
            chosen_logprobs, rejected_logprobs = all_logprobs.chunk(2, dim=0)
            ref_all_logprobs = _get_sequence_logprobs(ref_model, combined_input_ids, combined_attention_mask, combined_labels)
            ref_chosen_logprobs, ref_rejected_logprobs = ref_all_logprobs.chunk(2, dim=0)
            loss, chosen_rewards, rejected_rewards = _compute_dpo_loss(
                chosen_logprobs, rejected_logprobs,
                ref_chosen_logprobs, ref_rejected_logprobs,
                beta=config.beta,
            )

        _sync_if_cuda(device)
        elapsed = time.perf_counter() - start

        tracker.update({
            "dpo_loss": float(loss.item()),
            "chosen_logprob": float(chosen_logprobs.mean().item()),
            "rejected_logprob": float(rejected_logprobs.mean().item()),
            "chosen_reward": float(chosen_rewards.mean().item()),
            "rejected_reward": float(rejected_rewards.mean().item()),
            "preference_correct": int((chosen_rewards > rejected_rewards).sum().item()),
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "forward_time_sec": elapsed,
            "backward_time_sec": 0.0,
            "optimizer_time_sec": 0.0,
            "step_time_sec": elapsed,
        })

    metrics = tracker.compute()
    metrics["split"] = "val"
    return metrics
