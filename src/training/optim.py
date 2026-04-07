from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("name", "adamw")).lower()
    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(config.get("betas", (0.9, 0.999))),
        )

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(config.get("nesterov", False)),
        )

    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=weight_decay,
            alpha=float(config.get("alpha", 0.99)),
        )

    if name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError as exc:
            raise ImportError(
                "Optimizer 'lion' requires the optional package 'lion-pytorch'. "
                "Install it in your environment before using this optimizer."
            ) from exc
        return Lion(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(config.get("betas", (0.9, 0.99))),
        )

    raise ValueError(f"Unsupported optimizer '{name}'.")
