"""Keypoint definitions and utilities for sign language pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

BODY_KEYPOINTS: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

LEFT_HAND_KEYPOINTS: List[str] = [
    "left_hand_wrist",
    "left_hand_thumb_cmc",
    "left_hand_thumb_mcp",
    "left_hand_thumb_ip",
    "left_hand_thumb_tip",
    "left_hand_index_mcp",
    "left_hand_index_pip",
    "left_hand_index_dip",
    "left_hand_index_tip",
    "left_hand_middle_mcp",
    "left_hand_middle_pip",
    "left_hand_middle_dip",
    "left_hand_middle_tip",
    "left_hand_ring_mcp",
    "left_hand_ring_pip",
    "left_hand_ring_dip",
    "left_hand_ring_tip",
    "left_hand_little_mcp",
    "left_hand_little_pip",
    "left_hand_little_dip",
    "left_hand_little_tip",
]

RIGHT_HAND_KEYPOINTS: List[str] = [
    name.replace("left_", "right_") for name in LEFT_HAND_KEYPOINTS
]

FACE_KEYPOINTS: List[str] = [
    "face_nose",
    "face_left_eye",
    "face_right_eye",
    "face_left_ear",
    "face_right_ear",
]

KEYPOINT_NAMES: List[str] = BODY_KEYPOINTS + LEFT_HAND_KEYPOINTS + RIGHT_HAND_KEYPOINTS + FACE_KEYPOINTS

INDEX_BY_NAME: Dict[str, int] = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}
NAME_BY_INDEX: Dict[int, str] = {idx: name for name, idx in INDEX_BY_NAME.items()}

LEFT_RIGHT_PAIRS: List[Tuple[str, str]] = [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_hand_wrist", "right_hand_wrist"),
    ("left_hand_thumb_cmc", "right_hand_thumb_cmc"),
    ("left_hand_thumb_mcp", "right_hand_thumb_mcp"),
    ("left_hand_thumb_ip", "right_hand_thumb_ip"),
    ("left_hand_thumb_tip", "right_hand_thumb_tip"),
    ("left_hand_index_mcp", "right_hand_index_mcp"),
    ("left_hand_index_pip", "right_hand_index_pip"),
    ("left_hand_index_dip", "right_hand_index_dip"),
    ("left_hand_index_tip", "right_hand_index_tip"),
    ("left_hand_middle_mcp", "right_hand_middle_mcp"),
    ("left_hand_middle_pip", "right_hand_middle_pip"),
    ("left_hand_middle_dip", "right_hand_middle_dip"),
    ("left_hand_middle_tip", "right_hand_middle_tip"),
    ("left_hand_ring_mcp", "right_hand_ring_mcp"),
    ("left_hand_ring_pip", "right_hand_ring_pip"),
    ("left_hand_ring_dip", "right_hand_ring_dip"),
    ("left_hand_ring_tip", "right_hand_ring_tip"),
    ("left_hand_little_mcp", "right_hand_little_mcp"),
    ("left_hand_little_pip", "right_hand_little_pip"),
    ("left_hand_little_dip", "right_hand_little_dip"),
    ("left_hand_little_tip", "right_hand_little_tip"),
    ("face_left_eye", "face_right_eye"),
    ("face_left_ear", "face_right_ear"),
]

SKELETON: List[Tuple[str, str]] = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hand_wrist", "left_wrist"),
    ("right_hand_wrist", "right_wrist"),
]

# Hand skeleton edges per finger
HAND_FINGER_CONNECTIONS: List[Tuple[str, str]] = []
for prefix in ("left", "right"):
    wrist = f"{prefix}_hand_wrist"
    for finger in ("thumb", "index", "middle", "ring", "little"):
        joints: Sequence[str]
        if finger == "thumb":
            joints = ["cmc", "mcp", "ip", "tip"]
        else:
            joints = ["mcp", "pip", "dip", "tip"]
        prev = wrist
        for joint in joints:
            name = f"{prefix}_hand_{finger}_{joint}"
            HAND_FINGER_CONNECTIONS.append((prev, name))
            prev = name

SKELETON.extend(HAND_FINGER_CONNECTIONS)
SKELETON.extend(
    [
        ("face_nose", "face_left_eye"),
        ("face_nose", "face_right_eye"),
        ("face_left_eye", "face_left_ear"),
        ("face_right_eye", "face_right_ear"),
    ]
)

@dataclass(frozen=True)
class KeypointInfo:
    """Convenience structure describing keypoint metadata."""

    name: str
    index: int


def all_keypoints() -> List[str]:
    """Returns the ordered list of keypoint names."""

    return KEYPOINT_NAMES.copy()


def to_index(name: str) -> int:
    """Looks up the index for a given keypoint name."""

    if name not in INDEX_BY_NAME:
        raise KeyError(f"Unknown keypoint name: {name}")
    return INDEX_BY_NAME[name]


def to_name(index: int) -> str:
    """Returns the keypoint name for a provided index."""

    if index not in NAME_BY_INDEX:
        raise KeyError(f"Invalid keypoint index: {index}")
    return NAME_BY_INDEX[index]


def left_right_pairs() -> List[Tuple[int, int]]:
    """Returns index pairs for horizontally flipping keypoints."""

    return [(INDEX_BY_NAME[l], INDEX_BY_NAME[r]) for l, r in LEFT_RIGHT_PAIRS]


def skeleton_edges() -> List[Tuple[int, int]]:
    """Returns index pairs representing the drawing skeleton."""

    edges: List[Tuple[int, int]] = []
    for left, right in SKELETON:
        if left in INDEX_BY_NAME and right in INDEX_BY_NAME:
            edges.append((INDEX_BY_NAME[left], INDEX_BY_NAME[right]))
    return edges


def iter_keypoint_info() -> Iterable[KeypointInfo]:
    """Iterates over `KeypointInfo` entries describing each keypoint."""

    for name, index in INDEX_BY_NAME.items():
        yield KeypointInfo(name=name, index=index)
