from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .ade20k import ADE20KSegmentationConfig, build_ade20k_dataloaders
from .voc import VOCSegmentationConfig, build_voc_dataloaders


def _coerce_image_size(data_cfg: Mapping[str, Any]) -> tuple[int, int]:
    size = data_cfg.get("image_size", 256)
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    s = int(size)
    return s, s


def _resolve_data_root(data_cfg: Mapping[str, Any], name: str, project_root: Path) -> Path:
    raw = data_cfg.get("root")
    if raw is not None and str(raw).strip() != "":
        return Path(raw).expanduser()
    if name == "ade20k":
        return project_root / "data" / "ADEChallengeData2016"
    return project_root / "data"


def build_segmentation_dataloaders(
    data_cfg: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """
    Build train/val dataloaders and ``class_names`` for a named segmentation dataset.

    ``data.name`` selects the dataset (default: ``voc``). Supported: ``voc``, ``ade20k``.

    If ``data.root`` is omitted, defaults are ``<project>/data`` for VOC and
    ``<project>/data/ADEChallengeData2016`` for ADE20K.
    """
    name = str(data_cfg.get("name", "voc")).lower()
    root = _resolve_data_root(data_cfg, name, project_root)
    image_size = _coerce_image_size(data_cfg)
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    download = bool(data_cfg.get("download", False))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    if name == "voc":
        return build_voc_dataloaders(
            VOCSegmentationConfig(
                data_root=str(root),
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                download=download,
                pin_memory=pin_memory,
            )
        )
    if name == "ade20k":
        return build_ade20k_dataloaders(
            ADE20KSegmentationConfig(
                data_root=str(root),
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                download=download,
                pin_memory=pin_memory,
            )
        )
    raise ValueError(f"Unknown data.name {name!r}. Supported: 'voc', 'ade20k'.")
