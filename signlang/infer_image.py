"""Single image inference script."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

import yaml

from . import metrics, model, utils, viz

LOGGER = logging.getLogger("signlang.infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Optional config to override model settings")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.2)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def preprocess(image_path: Path, img_size: int) -> Tuple[np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return original, tensor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    cfg = None
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if cfg and "training" in cfg:
            args.img_size = int(cfg["training"].get("img_size", args.img_size))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    original_image, input_tensor = preprocess(Path(args.image), args.img_size)
    input_tensor = input_tensor.to(device)

    model_cfg = cfg if cfg else {"model": {"backbone": "resnet18"}}
    model_module = model.build_model(model_cfg).to(device)
    checkpoint = utils.load_checkpoint(Path(args.checkpoint), map_location=device_str)
    model_module.load_state_dict(checkpoint["model_state"])
    model_module.eval()

    with torch.no_grad():
        outputs = model_module(input_tensor)
    coords_heatmap, confidences = metrics.decode_heatmaps(outputs.cpu())
    heatmap_size = outputs.shape[-1]
    scale = args.img_size / heatmap_size
    coords = coords_heatmap[0].numpy() * scale
    confidences = confidences[0].numpy()

    visibility = (confidences >= args.conf_threshold).astype(np.float32)
    h, w, _ = original_image.shape
    coords[:, 0] *= w / args.img_size
    coords[:, 1] *= h / args.img_size
    overlay = viz.draw_skeleton(original_image, coords, visibility)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        LOGGER.info("Saved visualization to %s", out_path)
    if args.show:
        cv2.imshow("signlang", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
