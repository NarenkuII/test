"""Visualization utilities for sign language keypoints."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from . import keypoints as kp

Color = Tuple[int, int, int]


def draw_skeleton(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    visibility: np.ndarray,
    color: Color = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draws skeleton lines on a copy of the input image."""

    canvas = image.copy()
    for idx, (x, y) in enumerate(keypoints_xy):
        if visibility[idx] <= 0:
            continue
        cv2.circle(canvas, (int(x), int(y)), radius=3, color=color, thickness=-1)
    for start_idx, end_idx in kp.skeleton_edges():
        if visibility[start_idx] <= 0 or visibility[end_idx] <= 0:
            continue
        start = tuple(np.round(keypoints_xy[start_idx]).astype(int))
        end = tuple(np.round(keypoints_xy[end_idx]).astype(int))
        cv2.line(canvas, start, end, color=color, thickness=thickness)
    return canvas


def overlay_heatmaps(image: np.ndarray, heatmaps: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays averaged heatmap intensity onto the image."""

    avg_heatmap = np.mean(heatmaps, axis=0)
    norm = cv2.normalize(avg_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return blended
