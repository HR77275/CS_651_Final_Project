from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation

from .transforms import build_eval_transform, build_train_transform


VOC_CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


@dataclass
class VOCSegmentationConfig:
    data_root: str = "data"
    year: str = "2012"
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 4
    download: bool = False
    pin_memory: bool = True


class PascalVOCSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_set: str,
        image_size: tuple[int, int],
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_set = image_set
        self.dataset = VOCSegmentation(
            root=str(self.root),
            year="2012",
            image_set=image_set,
            download=download,
        )
        if image_set == "train":
            self.transform = build_train_transform(image_size)
        else:
            self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, mask = self.dataset[index]
        image = image.convert("RGB")
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "index": index,
        }


def build_voc_dataloaders(config: VOCSegmentationConfig) -> Dict[str, DataLoader]:
    train_dataset = PascalVOCSegmentationDataset(
        root=config.data_root,
        image_set="train",
        image_size=config.image_size,
        download=config.download,
    )
    val_dataset = PascalVOCSegmentationDataset(
        root=config.data_root,
        image_set="val",
        image_size=config.image_size,
        download=False,
    )

    common_loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "class_names": VOC_CLASSES,
    }

