from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import VOCSegmentationConfig, build_voc_dataloaders
from src.models import SegmentationModelConfig, build_segmentation_model
from src.training import StepConfig, train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DeepLabV3 training step on Pascal VOC.")
    parser.add_argument("--data-root", default="data", help="Pascal VOC root.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize for the sample batch.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Target device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    loaders = build_voc_dataloaders(
        VOCSegmentationConfig(
            data_root=args.data_root,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=False,
        )
    )
    batch = next(iter(loaders["train"]))

    model = build_segmentation_model(
        SegmentationModelConfig(
            name="deeplabv3_resnet50",
            num_classes=21,
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
