from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterable
import sys

import numpy as np
import torch
from torch.amp import GradScaler
from tqdm.auto import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import PreferenceDataConfig, build_preference_dataloaders
from src.models import LMConfig, build_lm_model
from src.training import DPOStepConfig, evaluate_dpo_epoch, train_dpo_epoch
from src.training.optim import build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO fine-tuning from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit_loader(loader: Iterable, max_batches: int | None):
    if max_batches is None:
        return loader

    class LimitedLoader:
        def __iter__(self_inner):
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                yield batch

        def __len__(self_inner):
            return min(len(loader), max_batches)

    return LimitedLoader()


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_serializable(record)) + "\n")


def _build_dpo_optimizer(policy, optimizer_cfg: dict) -> torch.optim.Optimizer:
    # Pass only trainable (LoRA adapter) params to avoid allocating optimizer
    # state for the frozen base-model weights.
    trainable = [p for p in policy.parameters() if p.requires_grad]
    name = str(optimizer_cfg.get("name", "adamw")).lower()
    lr = float(optimizer_cfg.get("lr", 5e-5))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    if name == "adamw":
        return torch.optim.AdamW(
            trainable, lr=lr, weight_decay=weight_decay,
            betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        )
    if name == "sgd":
        return torch.optim.SGD(
            trainable, lr=lr, momentum=float(optimizer_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(optimizer_cfg.get("nesterov", False)),
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            trainable, lr=lr, momentum=float(optimizer_cfg.get("momentum", 0.0)),
            weight_decay=weight_decay, alpha=float(optimizer_cfg.get("alpha", 0.99)),
        )
    if name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError as exc:
            raise ImportError("Install 'lion-pytorch' to use the Lion optimizer.") from exc
        return Lion(
            trainable, lr=lr, weight_decay=weight_decay,
            betas=tuple(optimizer_cfg.get("betas", (0.9, 0.99))),
        )
    raise ValueError(f"Unsupported optimizer '{name}'.")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    optimizer_cfg = config.get("optimizer", {})
    training_cfg = config.get("training", {})

    seed_everything(int(experiment_cfg.get("seed", 42)))

    requested_device = str(training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    # Model
    lm_config = LMConfig(
        model_name=str(model_cfg.get("model_name", "Qwen/Qwen2.5-0.5B")),
        use_lora=bool(model_cfg.get("use_lora", True)),
        lora_r=int(model_cfg.get("lora_r", 16)),
        lora_alpha=int(model_cfg.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=list(model_cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
    )
    policy, ref_model, tokenizer = build_lm_model(lm_config)
    if not lm_config.load_in_4bit:
        policy = policy.to(device)
    ref_model = ref_model.to(device)

    # Data
    data_config = PreferenceDataConfig(
        dataset_name=str(data_cfg.get("dataset_name", "argilla/ultrafeedback-binarized-preferences-cleaned")),
        tokenizer_name=str(data_cfg.get("tokenizer_name", lm_config.model_name)),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 256)),
        max_response_length=int(data_cfg.get("max_response_length", 512)),
        batch_size=int(data_cfg.get("batch_size", 4)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        max_train_samples=data_cfg.get("max_train_samples"),
        max_val_samples=data_cfg.get("max_val_samples"),
    )
    loaders = build_preference_dataloaders(data_config, tokenizer)

    # Optimizer + step config
    optimizer = _build_dpo_optimizer(policy, optimizer_cfg)
    step_config = DPOStepConfig(
        beta=float(training_cfg.get("beta", 0.1)),
        zero_grad_set_to_none=bool(training_cfg.get("zero_grad_set_to_none", True)),
        track_grad_norm=bool(training_cfg.get("track_grad_norm", True)),
        grad_norm_type=float(training_cfg.get("grad_norm_type", 2.0)),
        use_amp=bool(training_cfg.get("use_amp", False)),
    )
    scaler = GradScaler("cuda", enabled=bool(step_config.use_amp and device.type == "cuda"))

    # Output paths
    output_root = Path(experiment_cfg.get("output_dir", PROJECT_ROOT / "outputs"))
    run_name = str(experiment_cfg.get("name", "dpo_run"))
    run_dir = output_root / run_name
    if run_dir.exists():
        tqdm.write(f"[warn] Deleting existing run directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copy2(config_path, run_dir / "config.yaml")

    history_path = run_dir / "history.jsonl"
    best_adapter_path = run_dir / "best_adapter"
    last_adapter_path = run_dir / "last_adapter"

    # Save tokenizer once — needed when loading the adapter back.
    tokenizer.save_pretrained(run_dir / "tokenizer")

    epochs = int(training_cfg.get("epochs", 3))
    max_train_batches = training_cfg.get("max_train_batches")
    max_val_batches = training_cfg.get("max_val_batches")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loader = maybe_limit_loader(loaders["train"], max_train_batches)
        val_loader = maybe_limit_loader(loaders["val"], max_val_batches)

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        train_metrics = train_dpo_epoch(
            policy=policy, ref_model=ref_model, dataloader=train_bar,
            optimizer=optimizer, device=device, scaler=scaler, config=step_config,
        )

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)
        val_metrics = evaluate_dpo_epoch(
            policy=policy, ref_model=ref_model, dataloader=val_bar,
            device=device, config=step_config,
        )

        for m in (train_metrics, val_metrics):
            m["epoch"] = epoch
            m["learning_rate"] = float(optimizer.param_groups[0]["lr"])

        tqdm.write(
            " | ".join([
                f"epoch={epoch}/{epochs}",
                f"train_loss={train_metrics['mean_dpo_loss']:.4f}",
                f"val_loss={val_metrics['mean_dpo_loss']:.4f}",
                f"val_pref_acc={val_metrics['preference_accuracy']:.4f}",
                f"val_reward_margin={val_metrics['reward_margin']:.4f}",
                f"tokens/sec={train_metrics['tokens_per_sec']:.1f}",
                f"opt_state_mb={train_metrics['optimizer_state_bytes'] / (1024**2):.2f}",
            ])
        )

        save_jsonl(history_path, {"train": train_metrics, "val": val_metrics})

        # Save only the LoRA adapter weights (a few MB, not the full base model).
        # Load back with: PeftModel.from_pretrained(base_model, "<path>/last_adapter")
        policy.save_pretrained(last_adapter_path)
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict(),
             "train_metrics": train_metrics, "val_metrics": val_metrics},
            last_adapter_path / "trainer_state.pt",
        )

        if val_metrics["mean_dpo_loss"] < best_val_loss:
            best_val_loss = val_metrics["mean_dpo_loss"]
            policy.save_pretrained(best_adapter_path)
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict(),
                 "train_metrics": train_metrics, "val_metrics": val_metrics},
                best_adapter_path / "trainer_state.pt",
            )

    tqdm.write(f"Training complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
