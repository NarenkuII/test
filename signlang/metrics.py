"""Evaluation metrics for keypoint detection."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

__all__ = ["decode_heatmaps", "compute_pck", "compute_metrics"]


def decode_heatmaps(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decodes heatmaps to keypoint coordinates using argmax with sub-pixel refinement."""

    if heatmaps.ndim != 4:
        raise ValueError("Heatmaps must be 4-dimensional (B, K, H, W)")
    batch_size, num_keypoints, height, width = heatmaps.shape
    heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
    conf, idx = torch.max(heatmaps_flat, dim=2)
    y = (idx // width).float()
    x = (idx % width).float()

    # Sub-pixel refinement using gradients
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(y)
    padded = torch.nn.functional.pad(heatmaps, (1, 1, 1, 1), mode="replicate")
    idx_y = (y + 1).long()
    idx_x = (x + 1).long()
    batch_indices = torch.arange(batch_size, device=heatmaps.device)[:, None]
    keypoint_indices = torch.arange(num_keypoints, device=heatmaps.device)[None, :]
    gather = padded[batch_indices, keypoint_indices, idx_y, idx_x]
    left = padded[batch_indices, keypoint_indices, idx_y, idx_x - 1]
    right = padded[batch_indices, keypoint_indices, idx_y, idx_x + 1]
    top = padded[batch_indices, keypoint_indices, idx_y - 1, idx_x]
    bottom = padded[batch_indices, keypoint_indices, idx_y + 1, idx_x]
    dx = (right - left) / (2 * torch.abs(gather) + 1e-6)
    dy = (bottom - top) / (2 * torch.abs(gather) + 1e-6)
    x = x + dx
    y = y + dy

    coords = torch.stack([x, y], dim=-1)
    return coords, conf


def compute_pck(
    preds: torch.Tensor,
    targets: torch.Tensor,
    visibility: torch.Tensor,
    bboxes: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Computes Percentage of Correct Keypoints (PCK) metric."""

    if preds.shape != targets.shape:
        raise ValueError("Shape mismatch between predictions and targets")
    widths = bboxes[:, 2]
    heights = bboxes[:, 3]
    norm = torch.sqrt(widths * heights).clamp_min(1.0)
    distances = torch.norm(preds - targets, dim=-1)
    correct = (distances <= threshold * norm[:, None]).float() * visibility
    return correct.sum(dim=1) / visibility.sum(dim=1).clamp_min(1.0)


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    visibility: torch.Tensor,
    bboxes: torch.Tensor,
) -> Dict[str, float]:
    """Aggregates PCK and MSE style metrics."""

    mse = ((preds - targets) ** 2).mean().item()
    pck_05 = compute_pck(preds, targets, visibility, bboxes, threshold=0.05).mean().item()
    pck_10 = compute_pck(preds, targets, visibility, bboxes, threshold=0.1).mean().item()
    oks = torch.exp(-((preds - targets) ** 2).sum(dim=-1) / (2 * (0.1 * torch.sqrt(bboxes[:, 2] * bboxes[:, 3]).unsqueeze(1)) ** 2 + 1e-6))
    oks = (oks * visibility).sum(dim=1) / visibility.sum(dim=1).clamp_min(1.0)
    return {"mse": mse, "pck@0.05": pck_05, "pck@0.1": pck_10, "oks_like": oks.mean().item()}
