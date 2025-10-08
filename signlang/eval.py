"""Evaluation script for sign language keypoint detector."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from . import data, losses, model, train, utils

LOGGER = logging.getLogger("signlang.eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sign language keypoint detector")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--labels-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    cfg = train.load_config(Path(args.config) if args.config else None)
    cfg = train.apply_cli_overrides(cfg, args)
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    utils.set_seed(train_cfg.get("seed", 42))

    device_str = args.device or train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    LOGGER.info("Evaluating on device: %s", device)

    _, val_dataset = train.create_datasets(cfg)
    val_loader = data.build_dataloaders(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=False,
    )

    model_module = model.build_model(cfg).to(device)
    checkpoint = utils.load_checkpoint(Path(args.checkpoint), map_location=device_str)
    model_module.load_state_dict(checkpoint["model_state"])
    criterion = losses.build_loss(cfg).to(device)

    metrics_dict = train.evaluate(val_loader, model_module, criterion, device, cfg)
    LOGGER.info("Evaluation metrics: %s", metrics_dict)


if __name__ == "__main__":
    main()
