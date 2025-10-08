"""Dataset and data loading utilities for sign language keypoints."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import keypoints as kp

__all__ = ["SignLanguageKeypoints", "build_dataloaders", "load_annotation"]


def load_annotation(path: Path) -> Dict[str, Any]:
    """Loads and validates a JSON annotation file."""

    with path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    required_fields = {"image", "width", "height", "keypoints"}
    missing = required_fields - set(annotation)
    if missing:
        raise ValueError(f"Annotation {path} missing fields: {missing}")

    if len(annotation["keypoints"]) != len(kp.KEYPOINT_NAMES):
        raise ValueError(
            f"Annotation {path} has {len(annotation['keypoints'])} keypoints; expected {len(kp.KEYPOINT_NAMES)}"
        )

    return annotation


def _build_augmentation_pipeline(config: Dict[str, Any], img_size: int, training: bool) -> A.Compose:
    """Creates an Albumentations augmentation pipeline."""

    transforms: List[A.BasicTransform] = []
    if not training or not config.get("enabled", True):
        transforms.append(A.Resize(height=img_size, width=img_size))
        transforms.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=["keypoint_labels"]),
        )
    if config.get("rotation", True):
        transforms.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.25, rotate_limit=30, border_mode=cv2.BORDER_REFLECT))
    if config.get("scale", True):
        transforms.append(A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.75, 1.25), ratio=(0.9, 1.1), interpolation=cv2.INTER_LINEAR))
    else:
        transforms.append(A.Resize(height=img_size, width=img_size))
    if config.get("color_jitter", True):
        transforms.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.8))
    if config.get("blur", False):
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=0.3))
    if config.get("noise", False):
        transforms.append(A.GaussNoise(var_limit=(5.0, 25.0), p=0.2))

    transforms.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=["keypoint_labels"]),
    )


@dataclass
class Sample:
    """Represents a single dataset element after preprocessing."""

    image: torch.Tensor
    heatmaps: torch.Tensor
    keypoints: torch.Tensor
    visibility: torch.Tensor
    meta: Dict[str, Any]


class SignLanguageKeypoints(Dataset):
    """PyTorch dataset for sign language keypoint detection."""

    def __init__(
        self,
        data_root: str | Path,
        images_dir: str | Path,
        labels_dir: str | Path,
        img_size: int = 512,
        heatmap_size: int = 128,
        sigma: float = 2.0,
        augmentations: Optional[Dict[str, Any]] = None,
        training: bool = True,
        file_stems: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / images_dir
        self.labels_dir = self.data_root / labels_dir
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.training = training
        self.augmentations_config = augmentations or {}

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.entries: List[Tuple[Path, Path]] = self._discover_entries(file_stems)
        if not self.entries:
            raise RuntimeError(f"No matching image/label pairs found in {self.data_root}")

        self.flip_pairs = kp.left_right_pairs()
        self.augmentation = _build_augmentation_pipeline(self.augmentations_config, self.img_size, training=self.training)

    def _discover_entries(self, file_stems: Optional[Sequence[str]]) -> List[Tuple[Path, Path]]:
        image_files = sorted(self.images_dir.glob("*"))
        entries: List[Tuple[Path, Path]] = []
        stem_filter = set(file_stems) if file_stems is not None else None
        seen_stems = set()
        for image_path in image_files:
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if stem_filter is not None and image_path.stem not in stem_filter:
                continue
            label_path = self.labels_dir / (image_path.stem + ".json")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file for {image_path.name}")
            entries.append((image_path, label_path))
            seen_stems.add(image_path.stem)
        if stem_filter is not None:
            missing = set(stem_filter) - seen_stems
            if missing:
                raise FileNotFoundError(f"Missing images for stems: {sorted(missing)}")
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image

    def _apply_horizontal_flip(self, image: np.ndarray, keypoints_xyv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = np.ascontiguousarray(image[:, ::-1, :])
        keypoints_xyv[:, 0] = image.shape[1] - keypoints_xyv[:, 0]
        for left_idx, right_idx in self.flip_pairs:
            keypoints_xyv[[left_idx, right_idx]] = keypoints_xyv[[right_idx, left_idx]]
        return image, keypoints_xyv

    def __getitem__(self, index: int) -> Sample:
        image_path, label_path = self.entries[index]
        annotation = load_annotation(label_path)
        image = self._load_image(image_path)
        h, w, _ = image.shape

        if annotation["width"] != w or annotation["height"] != h:
            raise ValueError(
                f"Annotation size mismatch for {image_path.name}: "
                f"expected ({annotation['width']}, {annotation['height']}), got ({w}, {h})"
            )

        keypoints_xy = np.zeros((len(kp.KEYPOINT_NAMES), 2), dtype=np.float32)
        visibility = np.zeros((len(kp.KEYPOINT_NAMES),), dtype=np.float32)
        labels: List[str] = []
        for entry in annotation["keypoints"]:
            name = entry["name"]
            if name not in kp.INDEX_BY_NAME:
                raise KeyError(f"Unknown keypoint '{name}' in {label_path}")
            idx = kp.INDEX_BY_NAME[name]
            x, y = float(entry["x"]), float(entry["y"])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"Non-finite coordinate for {name} in {label_path}")
            if not (-1e-3 <= x <= w + 1e-3 and -1e-3 <= y <= h + 1e-3):
                raise ValueError(f"Coordinate out of bounds for {name} in {label_path}: ({x}, {y})")
            keypoints_xy[idx] = (x, y)
            visibility[idx] = float(entry.get("visible", 1))
            labels.append(name)

        if len(labels) != len(kp.KEYPOINT_NAMES):
            raise ValueError(f"Annotation {label_path} missing keypoints; expected {len(kp.KEYPOINT_NAMES)}")

        if (
            self.training
            and self.augmentations_config.get("enabled", True)
            and self.augmentations_config.get("horizontal_flip", True)
        ):
            if random.random() < 0.5:
                image, keypoints_xyv = self._apply_horizontal_flip(
                    image, np.concatenate([keypoints_xy, visibility[:, None]], axis=1)
                )
                keypoints_xy = keypoints_xyv[:, :2]
                visibility = keypoints_xyv[:, 2]

        augmented = self.augmentation(image=image, keypoints=keypoints_xy.tolist(), keypoint_labels=kp.KEYPOINT_NAMES)
        image = augmented["image"]
        aug_keypoints = np.array(augmented["keypoints"], dtype=np.float32)
        aug_xy = aug_keypoints[:, :2]
        aug_visibility = visibility.copy()

        # Ensure coordinates within bounds after augmentation
        aug_xy = np.clip(aug_xy, 0, self.img_size - 1)

        heatmaps = self._generate_heatmaps(aug_xy, aug_visibility)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        heatmap_tensor = torch.from_numpy(heatmaps).float()
        keypoints_tensor = torch.from_numpy(aug_xy)
        visibility_tensor = torch.from_numpy(aug_visibility)

        meta = {
            "image_path": str(image_path),
            "label_path": str(label_path),
            "bbox": annotation.get("bbox", None),
            "original_size": (h, w),
        }

        return Sample(
            image=image_tensor,
            heatmaps=heatmap_tensor,
            keypoints=keypoints_tensor,
            visibility=visibility_tensor,
            meta=meta,
        )

    def _generate_heatmaps(self, keypoints_xy: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        num_keypoints = keypoints_xy.shape[0]
        heatmaps = np.zeros((num_keypoints, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        tmp_size = self.sigma * 3
        for idx in range(num_keypoints):
            if visibility[idx] <= 0:
                continue
            mu_x = keypoints_xy[idx, 0] * self.heatmap_size / self.img_size
            mu_y = keypoints_xy[idx, 1] * self.heatmap_size / self.img_size

            x_min = int(max(0, math.floor(mu_x - tmp_size)))
            x_max = int(min(self.heatmap_size - 1, math.ceil(mu_x + tmp_size)))
            y_min = int(max(0, math.floor(mu_y - tmp_size)))
            y_max = int(min(self.heatmap_size - 1, math.ceil(mu_y + tmp_size)))

            if x_max < x_min or y_max < y_min:
                continue

            xs = np.arange(x_min, x_max + 1, dtype=np.float32)
            ys = np.arange(y_min, y_max + 1, dtype=np.float32)[:, None]
            exponent = ((xs - mu_x) ** 2 + (ys - mu_y) ** 2) / (2 * self.sigma ** 2)
            heatmaps[idx, y_min : y_max + 1, x_min : x_max + 1] = np.exp(-exponent)
        return heatmaps


def collate_fn(batch: Sequence[Sample]) -> Dict[str, torch.Tensor]:
    images = torch.stack([sample.image for sample in batch])
    heatmaps = torch.stack([sample.heatmaps for sample in batch])
    keypoints = torch.stack([sample.keypoints for sample in batch])
    visibility = torch.stack([sample.visibility for sample in batch])
    return {
        "images": images,
        "heatmaps": heatmaps,
        "keypoints": keypoints,
        "visibility": visibility,
        "meta": [sample.meta for sample in batch],
    }


def build_dataloaders(
    dataset: SignLanguageKeypoints,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    """Returns a DataLoader for the dataset."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
