"""Realtime webcam demo for sign language keypoint detector."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from . import metrics, model, utils, viz

LOGGER = logging.getLogger("signlang.live")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def preprocess(frame: np.ndarray, img_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor, rgb


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    cfg = None
    if args.config:
        import yaml

        with Path(args.config).open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if cfg and "training" in cfg:
            args.img_size = int(cfg["training"].get("img_size", args.img_size))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model_cfg = cfg if cfg else {"model": {"backbone": "resnet18"}}
    model_module = model.build_model(model_cfg).to(device)
    checkpoint = utils.load_checkpoint(Path(args.checkpoint), map_location=device_str)
    model_module.load_state_dict(checkpoint["model_state"])
    model_module.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                LOGGER.warning("Failed to read frame from camera")
                break
            input_tensor, rgb = preprocess(frame, args.img_size)
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                outputs = model_module(input_tensor)
            coords_heatmap, confidences = metrics.decode_heatmaps(outputs.cpu())
            heatmap_size = outputs.shape[-1]
            scale = args.img_size / heatmap_size
            coords = coords_heatmap[0].numpy() * scale
            conf = confidences[0].numpy()
            visibility = (conf >= args.conf_threshold).astype(np.float32)
            h, w, _ = rgb.shape
            coords[:, 0] *= w / args.img_size
            coords[:, 1] *= h / args.img_size
            overlay = viz.draw_skeleton(rgb, coords, visibility)
            display = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imshow("signlang", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
