from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch.amp import GradScaler
from tqdm.auto import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_segmentation_dataloaders
from src.models import SegmentationModelConfig, build_segmentation_model
from src.training import StepConfig, build_segmentation_loss, build_optimizer, evaluate_epoch, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3 on a segmentation dataset from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
            for index, batch in enumerate(loader):
                if index >= max_batches:
                    break
                yield batch

        def __len__(self_inner):
            return min(len(loader), max_batches)

    return LimitedLoader()


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(record)) + "\n")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    optimizer_cfg = config.get("optimizer", {})
    training_cfg = config.get("training", {})

    seed = int(experiment_cfg.get("seed", 42))
    seed_everything(seed)

    requested_device = str(training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    loaders = build_segmentation_dataloaders(data_cfg, project_root=PROJECT_ROOT)
    class_names = loaders["class_names"]
    num_classes = len(class_names)

    model = build_segmentation_model(
        SegmentationModelConfig(
            name=str(model_cfg.get("name", "deeplabv3_resnet50")),
            num_classes=int(model_cfg.get("num_classes", num_classes)),
            pretrained_backbone=bool(model_cfg.get("pretrained_backbone", True)),
            aux_loss=bool(model_cfg.get("aux_loss", True)),
        )
    ).to(device)

    optimizer = build_optimizer(model, optimizer_cfg)
    step_config = StepConfig(
        ignore_index=int(training_cfg.get("ignore_index", 255)),
        aux_weight=float(training_cfg.get("aux_weight", 0.4)),
        zero_grad_set_to_none=bool(training_cfg.get("zero_grad_set_to_none", True)),
        track_grad_norm=bool(training_cfg.get("track_grad_norm", True)),
        grad_norm_type=float(training_cfg.get("grad_norm_type", 2.0)),
        use_amp=bool(training_cfg.get("use_amp", False)),
    )
    scaler = GradScaler("cuda", enabled=bool(step_config.use_amp and device.type == "cuda"))
    criterion = build_segmentation_loss(ignore_index=step_config.ignore_index)

    output_root = Path(experiment_cfg.get("output_dir", PROJECT_ROOT / "outputs"))
    run_name = str(experiment_cfg.get("name", "deeplabv3_seg_run"))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")

    history_path = run_dir / "history.jsonl"
    best_ckpt_path = run_dir / "best.pt"
    last_ckpt_path = run_dir / "last.pt"
    epochs = int(training_cfg.get("epochs", 10))
    max_train_batches = training_cfg.get("max_train_batches")
    max_val_batches = training_cfg.get("max_val_batches")

    best_miou = float("-inf")

    for epoch in range(1, epochs + 1):
        train_loader = maybe_limit_loader(loaders["train"], max_train_batches)
        val_loader = maybe_limit_loader(loaders["val"], max_val_batches)

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        train_metrics = train_epoch(
            model=model,
            dataloader=train_bar,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            criterion=criterion,
            scaler=scaler,
            class_names=class_names,
            config=step_config,
        )

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)
        val_metrics = evaluate_epoch(
            model=model,
            dataloader=val_bar,
            device=device,
            num_classes=num_classes,
            criterion=criterion,
            class_names=class_names,
            config=step_config,
        )

        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch
        train_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        val_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])

        tqdm.write(
            " | ".join(
                [
                    f"epoch={epoch}/{epochs}",
                    f"train_loss={train_metrics['mean_loss']:.4f}",
                    f"train_miou={train_metrics['miou']:.4f}",
                    f"val_loss={val_metrics['mean_loss']:.4f}",
                    f"val_miou={val_metrics['miou']:.4f}",
                    f"val_pixacc={val_metrics['pixel_accuracy']:.4f}",
                    f"opt_state_mb={train_metrics['optimizer_state_bytes'] / (1024 ** 2):.2f}",
                ]
            )
        )

        save_jsonl(history_path, {"train": train_metrics, "val": val_metrics})

        checkpoint = {
            "epoch": epoch,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, last_ckpt_path)

        if val_metrics["miou"] > best_miou:
            best_miou = float(val_metrics["miou"])
            torch.save(checkpoint, best_ckpt_path)

    tqdm.write(f"Training complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()

