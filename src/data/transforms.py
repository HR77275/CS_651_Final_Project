from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as F


@dataclass
class SegmentationPairTransform:
    resize: Tuple[int, int]
    horizontal_flip_prob: float = 0.0
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = F.resize(image, self.resize, interpolation=Image.BILINEAR)
        mask = F.resize(mask, self.resize, interpolation=Image.NEAREST)

        if self.horizontal_flip_prob > 0.0 and torch.rand(1).item() < self.horizontal_flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)

        mask_tensor = torch.as_tensor(F.pil_to_tensor(mask), dtype=torch.int64).squeeze(0)
        return image_tensor, mask_tensor


def build_train_transform(image_size: Tuple[int, int]) -> Callable:
    return SegmentationPairTransform(resize=image_size, horizontal_flip_prob=0.5)


def build_eval_transform(image_size: Tuple[int, int]) -> Callable:
    return SegmentationPairTransform(resize=image_size, horizontal_flip_prob=0.0)
