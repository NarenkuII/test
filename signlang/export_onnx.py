"""Export trained model to ONNX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from . import model, utils

LOGGER = logging.getLogger("signlang.export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model_module = model.build_model({"model": {"backbone": "resnet18"}}).to(device)
    checkpoint = utils.load_checkpoint(Path(args.checkpoint), map_location=device_str)
    model_module.load_state_dict(checkpoint["model_state"])
    model_module.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model_module,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        input_names=["images"],
        output_names=["heatmaps"],
        dynamic_axes={"images": {0: "batch"}, "heatmaps": {0: "batch"}},
    )
    LOGGER.info("Exported ONNX model to %s", output_path)


if __name__ == "__main__":
    main()
