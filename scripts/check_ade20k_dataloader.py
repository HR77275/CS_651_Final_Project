from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_segmentation_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check the ADE20K dataloader (ADEChallengeData2016 layout).")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to ADEChallengeData2016 (contains images/ and annotations/). Default: <project>/data/ADEChallengeData2016.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the check.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--image-size", type=int, default=256, help="Resize images and masks to a square shape.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract ADEChallengeData2016 if it is not present (large download).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: dict = {
        "name": "ade20k",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "download": args.download,
        "pin_memory": False,
    }
    if args.data_root is not None:
        cfg["root"] = args.data_root

    loaders = build_segmentation_dataloaders(cfg, project_root=PROJECT_ROOT)
    batch = next(iter(loaders["train"]))
    print("train_images:", tuple(batch["image"].shape))
    print("train_masks:", tuple(batch["mask"].shape))
    print("mask_min_max:", int(batch["mask"].min()), int(batch["mask"].max()))
    print("num_classes (names):", len(loaders["class_names"]))
    print("num_train_batches:", len(loaders["train"]))
    print("num_val_batches:", len(loaders["val"]))


if __name__ == "__main__":
    main()
