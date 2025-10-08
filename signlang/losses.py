"""Loss functions for heatmap regression."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

__all__ = ["HeatmapMSELoss", "FocalMSELoss", "build_loss"]


class HeatmapMSELoss(nn.Module):
    """Mean squared error on heatmaps with optional visibility masking."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction="none")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.criterion(pred, target)
        if visibility is not None:
            mask = visibility[:, :, None, None]
            loss = loss * mask
        if self.reduction == "mean":
            denom = loss.numel() if visibility is None else mask.sum().clamp_min(1.0)
            return loss.sum() / denom
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalMSELoss(HeatmapMSELoss):
    """Focal variant of the MSE loss to emphasize hard examples."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__(reduction="mean")
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = (pred - target) ** 2
        focal_weight = torch.pow(torch.abs(pred - target) + 1e-6, self.gamma)
        loss = loss * focal_weight
        if visibility is not None:
            mask = visibility[:, :, None, None]
            loss = loss * mask
            denom = mask.sum().clamp_min(1.0)
            return loss.sum() / denom
        return loss.mean()


def build_loss(config: Dict[str, object]) -> nn.Module:
    loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
    name = str(loss_cfg.get("name", "mse")).lower()
    if name == "mse":
        return HeatmapMSELoss()
    if name == "focal_mse":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        return FocalMSELoss(gamma=gamma)
    raise KeyError(f"Unsupported loss: {name}")
