from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_segmentation_dataloaders
from src.models import SegmentationModelConfig, build_segmentation_model
from src.training import StepConfig, train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DeepLabV3 training step on a segmentation dataset.")
    parser.add_argument("--dataset", choices=("voc", "ade20k"), default="voc", help="Segmentation dataset.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root (VOC: VOC parent folder; ADE20K: ADEChallengeData2016). Default: project data layout.",
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Override class count (default: 21 VOC, 151 ADE20K).")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize for the sample batch.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Target device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    data_cfg: dict = {
        "name": args.dataset,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "download": False,
        "pin_memory": False,
    }
    if args.data_root is not None:
        data_cfg["root"] = args.data_root

    loaders = build_segmentation_dataloaders(data_cfg, project_root=PROJECT_ROOT)
    batch = next(iter(loaders["train"]))

    num_classes = args.num_classes or (151 if args.dataset == "ade20k" else 21)
    model = build_segmentation_model(
        SegmentationModelConfig(
            name="deeplabv3_resnet50",
            num_classes=num_classes,
            pretrained_backbone=True,
            aux_loss=True,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    metrics = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        config=StepConfig(),
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
