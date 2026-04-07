from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import VOCSegmentationConfig, build_voc_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check the Pascal VOC dataloader.")
    parser.add_argument("--data-root", default="data", help="Root directory for Pascal VOC data.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the check.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--image-size", type=int, default=256, help="Resize images and masks to a square shape.")
    parser.add_argument("--download", action="store_true", help="Download VOC 2012 if it is not present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VOCSegmentationConfig(
        data_root=str(Path(args.data_root)),
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    loaders = build_voc_dataloaders(config)
    batch = next(iter(loaders["train"]))
    print("train_images:", tuple(batch["image"].shape))
    print("train_masks:", tuple(batch["mask"].shape))
    print("mask_classes_present:", sorted(batch["mask"].unique().tolist())[:20])
    print("num_train_batches:", len(loaders["train"]))
    print("num_val_batches:", len(loaders["val"]))


if __name__ == "__main__":
    main()
