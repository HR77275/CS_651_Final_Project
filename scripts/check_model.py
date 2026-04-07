from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import SegmentationModelConfig, available_segmentation_models, build_segmentation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instantiate and sanity-check a segmentation model.")
    parser.add_argument("--model", default="deeplabv3_resnet50", help="Model name to instantiate.")
    parser.add_argument("--image-size", type=int, default=256, help="Square input size for the dummy batch.")
    parser.add_argument("--batch-size", type=int, default=2, help="Dummy batch size.")
    parser.add_argument("--num-classes", type=int, default=21, help="Number of segmentation classes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("available_models:")
    for name, desc in available_segmentation_models().items():
        print(f"- {name}: {desc}")

    config = SegmentationModelConfig(name=args.model, num_classes=args.num_classes)
    model = build_segmentation_model(config)
    model.eval()

    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    with torch.no_grad():
        outputs = model(inputs)

    print("selected_model:", args.model)
    print("output_keys:", list(outputs.keys()))
    print("logit_shape:", tuple(outputs["out"].shape))
    if "aux" in outputs:
        print("aux_shape:", tuple(outputs["aux"].shape))


if __name__ == "__main__":
    main()
