from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from torch import nn
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)


MODEL_DESCRIPTIONS: Dict[str, str] = {
    "deeplabv3_resnet50": "Strong standard baseline with a ResNet-50 backbone.",
    "deeplabv3_mobilenet_v3_large": "Lighter DeepLabV3 variant that is faster to train.",
    "fcn_resnet50": "Older but simple fully convolutional baseline.",
    "lraspp_mobilenet_v3_large": "Lightweight mobile segmentation model for quick experiments.",
}


@dataclass
class SegmentationModelConfig:
    name: str = "deeplabv3_resnet50"
    num_classes: int = 21
    pretrained_backbone: bool = True
    aux_loss: bool = True


def available_segmentation_models() -> Dict[str, str]:
    return MODEL_DESCRIPTIONS.copy()


def _backbone_kwargs(pretrained_backbone: bool) -> Dict[str, object]:
    if pretrained_backbone:
        return {}
    return {"weights_backbone": None}


def build_segmentation_model(config: SegmentationModelConfig) -> nn.Module:
    if config.name == "deeplabv3_resnet50":
        return deeplabv3_resnet50(
            weights=None,
            num_classes=config.num_classes,
            aux_loss=config.aux_loss,
            **_backbone_kwargs(config.pretrained_backbone),
        )
    if config.name == "deeplabv3_mobilenet_v3_large":
        return deeplabv3_mobilenet_v3_large(
            weights=None,
            num_classes=config.num_classes,
            aux_loss=config.aux_loss,
            **_backbone_kwargs(config.pretrained_backbone),
        )
    if config.name == "fcn_resnet50":
        return fcn_resnet50(
            weights=None,
            num_classes=config.num_classes,
            aux_loss=config.aux_loss,
            **_backbone_kwargs(config.pretrained_backbone),
        )
    if config.name == "lraspp_mobilenet_v3_large":
        return lraspp_mobilenet_v3_large(
            weights=None,
            num_classes=config.num_classes,
            **_backbone_kwargs(config.pretrained_backbone),
        )
    valid = ", ".join(sorted(MODEL_DESCRIPTIONS))
    raise ValueError(f"Unsupported model '{config.name}'. Available models: {valid}")
