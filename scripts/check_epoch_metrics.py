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
from src.training import StepConfig, evaluate_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pascal VOC evaluation metrics for DeepLabV3.")
    parser.add_argument("--data-root", default="data", help="Pascal VOC root.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Target device.")
    parser.add_argument("--max-batches", type=int, default=2, help="Limit evaluation batches for the smoke test.")
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

    class LimitedLoader:
        def __init__(self, loader, max_batches: int):
            self.loader = loader
            self.max_batches = max_batches
        def __iter__(self):
            for idx, batch in enumerate(self.loader):
                if idx >= self.max_batches:
                    break
                yield batch

    model = build_segmentation_model(
        SegmentationModelConfig(
            name="deeplabv3_resnet50",
            num_classes=21,
            pretrained_backbone=True,
            aux_loss=True,
        )
    ).to(device)

    metrics = evaluate_epoch(
        model=model,
        dataloader=LimitedLoader(loaders["val"], args.max_batches),
        device=device,
        num_classes=21,
        class_names=loaders["class_names"],
        config=StepConfig(),
    )

    print("split:", metrics["split"])
    print("mean_loss:", metrics["mean_loss"])
    print("pixel_accuracy:", metrics["pixel_accuracy"])
    print("miou:", metrics["miou"])
    print("examples_per_sec:", metrics["examples_per_sec"])
    print("per_class_iou_named_sample:", list(metrics["per_class_iou_named"].items())[:5])


if __name__ == "__main__":
    main()
