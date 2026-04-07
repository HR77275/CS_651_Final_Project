from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_eval_transform, build_train_transform

# MIT Scene Parsing / ADE20K challenge archive (semantic segmentation).
ADE20K_CHALLENGE_ZIP_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

# 151 classes: index 0 = background, 1–150 = object/stuff categories (ADEChallengeData2016 masks).
ADE20K_NUM_CLASSES = 151


def load_ade20k_class_names() -> tuple[str, ...]:
    """Load human-readable class names (background + 150 ADE classes)."""
    path = Path(__file__).resolve().parent / "objectInfo150.txt"
    if not path.is_file():
        return tuple(["background"] + [f"class_{i}" for i in range(1, 151)])
    lines = path.read_text(encoding="utf-8").splitlines()
    names: List[str] = ["background"]
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            names.append(parts[4])
        else:
            # Fallback: space-separated, name is everything after Val column
            split = line.split()
            if len(split) >= 5:
                names.append(" ".join(split[4:]))
    if len(names) != ADE20K_NUM_CLASSES:
        return tuple(["background"] + [f"class_{i}" for i in range(1, 151)])
    return tuple(names)


ADE20K_CLASS_NAMES = load_ade20k_class_names()


def ensure_ade20k_downloaded(root: Path) -> None:
    """Download and extract ADEChallengeData2016 if ``root`` is missing."""
    root = Path(root)
    if (root / "images" / "training").is_dir() and (root / "annotations" / "training").is_dir():
        return
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    zip_path = parent / "ADEChallengeData2016.zip"
    if not zip_path.is_file():
        urllib.request.urlretrieve(ADE20K_CHALLENGE_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(parent)


def _load_semantic_mask(path: Path) -> Image.Image:
    """Load PNG mask as 8-bit grayscale preserving labels 0–150."""
    raw = Image.open(path)
    arr = np.asarray(raw, dtype=np.int64)
    arr = np.clip(arr, 0, 150)
    return Image.fromarray(arr.astype(np.uint8), mode="L")


@dataclass
class ADE20KSegmentationConfig:
    data_root: str = "data/ADEChallengeData2016"
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 4
    download: bool = False
    pin_memory: bool = True


class ADE20KSegmentationDataset(Dataset):
    """
    ADE20K scene parsing (ADEChallengeData2016 layout).

    Expected layout::

        {root}/images/{training|validation}/*.jpg
        {root}/annotations/{training|validation}/*.png
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: tuple[int, int],
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        if split not in ("training", "validation"):
            raise ValueError("split must be 'training' or 'validation'")
        if download:
            ensure_ade20k_downloaded(self.root)

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "annotations" / split
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"ADE20K images not found at {self.image_dir}. "
                f"Set data.download: true or extract ADEChallengeData2016 under {self.root}."
            )

        self._ids = sorted(
            p.stem for p in self.image_dir.glob("*.jpg") if (self.label_dir / f"{p.stem}.png").is_file()
        )
        if not self._ids:
            raise RuntimeError(f"No image/mask pairs found under {self.image_dir} and {self.label_dir}.")

        if split == "training":
            self.transform = build_train_transform(image_size)
        else:
            self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int):
        image_id = self._ids[index]
        image_path = self.image_dir / f"{image_id}.jpg"
        label_path = self.label_dir / f"{image_id}.png"
        image = Image.open(image_path).convert("RGB")
        mask = _load_semantic_mask(label_path)
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "index": index,
        }


def build_ade20k_dataloaders(config: ADE20KSegmentationConfig) -> Dict[str, Any]:
    train_dataset = ADE20KSegmentationDataset(
        root=config.data_root,
        split="training",
        image_size=config.image_size,
        download=config.download,
    )
    val_dataset = ADE20KSegmentationDataset(
        root=config.data_root,
        split="validation",
        image_size=config.image_size,
        download=False,
    )

    common_loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)

    return {
        "train": train_loader,
        "val": val_loader,
        "class_names": ADE20K_CLASS_NAMES,
    }
