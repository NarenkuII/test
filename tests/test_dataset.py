import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from signlang import data, keypoints


def create_dummy_dataset(root: Path) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[..., 0] = 255
    cv2.imwrite(str(images_dir / "000001.jpg"), image)
    keypoints_data = []
    for idx, name in enumerate(keypoints.KEYPOINT_NAMES):
        keypoints_data.append({"name": name, "x": float(5 + idx), "y": float(5 + idx), "visible": 1})
    annotation = {
        "image": "000001.jpg",
        "width": 512,
        "height": 512,
        "keypoints": keypoints_data,
        "bbox": [0, 0, 512, 512],
    }
    with (labels_dir / "000001.json").open("w", encoding="utf-8") as handle:
        json.dump(annotation, handle)


def test_dataset_loading(tmp_path: Path) -> None:
    create_dummy_dataset(tmp_path)
    dataset = data.SignLanguageKeypoints(
        data_root=str(tmp_path),
        images_dir="images",
        labels_dir="labels",
        img_size=512,
        heatmap_size=128,
        sigma=2.0,
        augmentations={"enabled": False},
        training=False,
    )
    sample = dataset[0]
    assert sample.image.shape == torch.Size([3, 512, 512])
    assert sample.heatmaps.shape == torch.Size([len(keypoints.KEYPOINT_NAMES), 128, 128])
    assert torch.all(sample.visibility == 1)


def test_collate_fn(tmp_path: Path) -> None:
    create_dummy_dataset(tmp_path)
    dataset = data.SignLanguageKeypoints(
        data_root=str(tmp_path),
        images_dir="images",
        labels_dir="labels",
        img_size=512,
        heatmap_size=128,
        sigma=2.0,
        augmentations={"enabled": False},
        training=False,
    )
    batch = data.collate_fn([dataset[0], dataset[0]])
    assert batch["images"].shape == torch.Size([2, 3, 512, 512])
    assert batch["heatmaps"].shape[0] == 2
