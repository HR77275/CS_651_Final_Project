from __future__ import annotations

from typing import Any, Dict


class DPOMetricTracker:
    def __init__(self) -> None:
        self.total_dpo_loss = 0.0
        self.total_batches = 0
        self.total_examples = 0
        self.total_tokens = 0
        self.total_preference_correct = 0
        self.total_chosen_logprob = 0.0
        self.total_rejected_logprob = 0.0
        self.total_chosen_reward = 0.0
        self.total_rejected_reward = 0.0
        self.total_step_time_sec = 0.0
        self.total_forward_time_sec = 0.0
        self.total_backward_time_sec = 0.0
        self.total_optimizer_time_sec = 0.0
        self.peak_gpu_allocated_bytes: int = 0

    def update(self, step_metrics: Dict[str, Any]) -> None:
        self.total_batches += 1
        self.total_examples += int(step_metrics.get("batch_size", 0))
        self.total_tokens += int(step_metrics.get("num_tokens", 0))
        self.total_dpo_loss += float(step_metrics.get("dpo_loss", 0.0))
        self.total_preference_correct += int(step_metrics.get("preference_correct", 0))
        self.total_chosen_logprob += float(step_metrics.get("chosen_logprob", 0.0))
        self.total_rejected_logprob += float(step_metrics.get("rejected_logprob", 0.0))
        self.total_chosen_reward += float(step_metrics.get("chosen_reward", 0.0))
        self.total_rejected_reward += float(step_metrics.get("rejected_reward", 0.0))
        self.total_step_time_sec += float(step_metrics.get("step_time_sec", 0.0))
        self.total_forward_time_sec += float(step_metrics.get("forward_time_sec", 0.0))
        self.total_backward_time_sec += float(step_metrics.get("backward_time_sec", 0.0))
        self.total_optimizer_time_sec += float(step_metrics.get("optimizer_time_sec", 0.0))
        gpu_peak = step_metrics.get("gpu_peak_allocated_bytes")
        if gpu_peak is not None:
            self.peak_gpu_allocated_bytes = max(self.peak_gpu_allocated_bytes, int(gpu_peak))

    def compute(self) -> Dict[str, Any]:
        n = max(self.total_batches, 1)
        total_time = max(self.total_step_time_sec, 1e-12)
        mean_chosen_logprob = self.total_chosen_logprob / n
        mean_rejected_logprob = self.total_rejected_logprob / n
        mean_chosen_reward = self.total_chosen_reward / n
        mean_rejected_reward = self.total_rejected_reward / n
        return {
            "mean_dpo_loss": self.total_dpo_loss / n,
            "preference_accuracy": float(self.total_preference_correct / max(self.total_examples, 1)),
            "reward_margin": mean_chosen_reward - mean_rejected_reward,
            "mean_chosen_logprob": mean_chosen_logprob,
            "mean_rejected_logprob": mean_rejected_logprob,
            "logprob_gap": mean_chosen_logprob - mean_rejected_logprob,
            "mean_chosen_reward": mean_chosen_reward,
            "mean_rejected_reward": mean_rejected_reward,
            "tokens_per_sec": float(self.total_tokens / total_time),
            "num_batches": self.total_batches,
            "num_examples": self.total_examples,
            "total_tokens": self.total_tokens,
            "epoch_time_sec": self.total_step_time_sec,
            "mean_step_time_sec": self.total_step_time_sec / n,
            "mean_forward_time_sec": self.total_forward_time_sec / n,
            "mean_backward_time_sec": self.total_backward_time_sec / n,
            "mean_optimizer_time_sec": self.total_optimizer_time_sec / n,
            "gpu_peak_allocated_bytes": self.peak_gpu_allocated_bytes or None,
        }
