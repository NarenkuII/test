"""Utility helpers for training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "AverageMeter",
    "ExponentialMovingAverage",
    "TensorBoardLogger",
]


def set_seed(seed: int) -> None:
    """Sets random seed across libraries for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - GPU dependent
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: Optional[str] = None) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


class AverageMeter:
    """Tracks running averages for scalar metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class ExponentialMovingAverage:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.original: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.original[name] = param.data.clone()
            param.data = self.shadow[name].clone()

    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.original:
                param.data = self.original[name].clone()
        self.original.clear()


@dataclass
class TensorBoardLogger:
    """Thin wrapper around SummaryWriter to ensure consistent usage."""

    log_dir: Path
    writer: SummaryWriter

    @classmethod
    def create(cls, log_dir: Path) -> "TensorBoardLogger":
        log_dir.mkdir(parents=True, exist_ok=True)
        return cls(log_dir=log_dir, writer=SummaryWriter(log_dir=str(log_dir)))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()
