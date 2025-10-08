"""Model definitions for sign language keypoint detection."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
from torch import nn
from torchvision import models

from . import keypoints as kp

__all__ = ["KeypointModel", "build_model", "BACKBONES"]


BACKBONES: Dict[str, Callable[[bool], Tuple[nn.Module, int, int]]] = {}


def _register_backbone(name: str) -> Callable[[Callable[[bool], Tuple[nn.Module, int, int]]], Callable[[bool], Tuple[nn.Module, int, int]]]:
    def decorator(func: Callable[[bool], Tuple[nn.Module, int, int]]) -> Callable[[bool], Tuple[nn.Module, int, int]]:
        BACKBONES[name] = func
        return func

    return decorator


@_register_backbone("resnet18")
def _resnet18(pretrained: bool) -> Tuple[nn.Module, int, int]:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = models.resnet18(weights=weights)
    modules = list(backbone.children())[:-2]
    backbone = nn.Sequential(*modules)
    return backbone, 512, 32


@_register_backbone("mobilenet_v3_small")
def _mobilenet_v3_small(pretrained: bool) -> Tuple[nn.Module, int, int]:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    backbone = models.mobilenet_v3_small(weights=weights).features
    return backbone, 576, 32


def build_head(in_channels: int, num_keypoints: int) -> nn.Module:
    """Constructs a simple deconvolutional upsampling head."""

    layers = []
    head_channels = [256, 256, 128]
    for out_channels in head_channels:
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.Conv2d(in_channels, num_keypoints, kernel_size=1))
    return nn.Sequential(*layers)


class KeypointModel(nn.Module):
    """Backbone + heatmap head network."""

    def __init__(
        self,
        num_keypoints: int = len(kp.KEYPOINT_NAMES),
        backbone: str = "resnet18",
        head_type: str = "deconv_upsample",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise KeyError(f"Unsupported backbone: {backbone}")
        if head_type != "deconv_upsample":
            raise KeyError(f"Unsupported head type: {head_type}")
        backbone_module, backbone_channels, stride = BACKBONES[backbone](pretrained)
        self.backbone = backbone_module
        self.backbone_stride = stride
        self.head = build_head(backbone_channels, num_keypoints)
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning heatmaps of shape (B, K, H, W)."""

        features = self.backbone(x)
        heatmaps = self.head(features)
        return heatmaps


def build_model(config: Dict[str, object]) -> KeypointModel:
    """Builds the model from a config dictionary."""

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    backbone = model_cfg.get("backbone", "resnet18")
    head_type = model_cfg.get("head", "deconv_upsample")
    pretrained = bool(model_cfg.get("pretrained", False))
    return KeypointModel(backbone=backbone, head_type=head_type, pretrained=pretrained)
