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
from src.training import StepConfig, evaluate_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation evaluation metrics for DeepLabV3.")
    parser.add_argument("--dataset", choices=("voc", "ade20k"), default="voc", help="Segmentation dataset.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root (VOC or ADEChallengeData2016). Default: project data layout.",
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Override class count (default: 21 VOC, 151 ADE20K).")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Target device.")
    parser.add_argument("--max-batches", type=int, default=2, help="Limit evaluation batches for the smoke test.")
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=None,
        help="Loss/metric ignore index (default: 255 VOC, -100 ADE20K).",
    )
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

    class LimitedLoader:
        def __init__(self, loader, max_batches: int):
            self.loader = loader
            self.max_batches = max_batches
        def __iter__(self):
            for idx, batch in enumerate(self.loader):
                if idx >= self.max_batches:
                    break
                yield batch

    num_classes = args.num_classes or (151 if args.dataset == "ade20k" else 21)
    ignore_index = args.ignore_index if args.ignore_index is not None else (-100 if args.dataset == "ade20k" else 255)

    model = build_segmentation_model(
        SegmentationModelConfig(
            name="deeplabv3_resnet50",
            num_classes=num_classes,
            pretrained_backbone=True,
            aux_loss=True,
        )
    ).to(device)

    metrics = evaluate_epoch(
        model=model,
        dataloader=LimitedLoader(loaders["val"], args.max_batches),
        device=device,
        num_classes=num_classes,
        class_names=loaders["class_names"],
        config=StepConfig(ignore_index=ignore_index),
    )

    print("split:", metrics["split"])
    print("mean_loss:", metrics["mean_loss"])
    print("pixel_accuracy:", metrics["pixel_accuracy"])
    print("miou:", metrics["miou"])
    print("examples_per_sec:", metrics["examples_per_sec"])
    print("per_class_iou_named_sample:", list(metrics["per_class_iou_named"].items())[:5])


if __name__ == "__main__":
    main()
