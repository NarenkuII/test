"""Training entry point for sign language keypoint detector."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader

from . import data, losses, metrics, model, utils

LOGGER = logging.getLogger("signlang.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sign language keypoint detector")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory containing dataset subfolders")
    parser.add_argument("--images-dir", type=str, default=None, help="Images subdirectory relative to data root")
    parser.add_argument("--labels-dir", type=str, default=None, help="Labels subdirectory relative to data root")
    parser.add_argument("--outdir", type=str, default=None, help="Directory for checkpoints and logs")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable AMP training")
    parser.add_argument("--device", type=str, default=None, help="Override torch device")
    return parser.parse_args()


def load_config(path: Path | None) -> Dict[str, Any]:
    default_path = Path(__file__).with_name("config.yaml")
    with default_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            user_cfg = yaml.safe_load(handle)
        cfg = _merge_dicts(cfg, user_cfg)
    return cfg


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    training_cfg = cfg.setdefault("training", {})
    model_cfg = cfg.setdefault("model", {})
    optimizer_cfg = cfg.setdefault("optimizer", {})
    paths_cfg = cfg.setdefault("paths", {})

    if args.data_root:
        paths_cfg["data_root"] = args.data_root
    if args.images_dir:
        paths_cfg["images_dir"] = args.images_dir
    if args.labels_dir:
        paths_cfg["labels_dir"] = args.labels_dir
    if args.outdir:
        paths_cfg["output_dir"] = args.outdir
    if args.epochs:
        training_cfg["epochs"] = args.epochs
    if args.batch_size:
        training_cfg["batch_size"] = args.batch_size
    if args.lr:
        optimizer_cfg["lr"] = args.lr
    if args.img_size:
        training_cfg["img_size"] = args.img_size
    if args.backbone:
        model_cfg["backbone"] = args.backbone
    if args.mixed_precision:
        training_cfg["mixed_precision"] = True
    if args.device:
        training_cfg["device"] = args.device
    if args.resume:
        training_cfg["resume"] = args.resume
    return cfg


def create_datasets(cfg: Dict[str, Any]) -> Tuple[data.SignLanguageKeypoints, data.SignLanguageKeypoints]:
    paths_cfg = cfg["paths"]
    training_cfg = cfg["training"]
    data_root = paths_cfg["data_root"]
    images_dir = paths_cfg["images_dir"]
    labels_dir = paths_cfg["labels_dir"]

    image_root = Path(data_root) / images_dir
    stems = sorted([p.stem for p in image_root.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not stems:
        raise RuntimeError(f"No image files found in {image_root}. Ensure dataset is prepared.")
    split_idx = max(int(len(stems) * 0.9), 1)
    train_stems = stems[:split_idx]
    val_stems = stems[split_idx:] if split_idx < len(stems) else stems[-1:]

    augmentations = cfg.get("augmentations", {})
    train_dataset = data.SignLanguageKeypoints(
        data_root=data_root,
        images_dir=images_dir,
        labels_dir=labels_dir,
        img_size=training_cfg["img_size"],
        heatmap_size=training_cfg["heatmap_size"],
        sigma=training_cfg["sigma"],
        augmentations=augmentations,
        training=True,
        file_stems=train_stems,
    )
    val_aug = dict(augmentations)
    val_aug["enabled"] = False
    val_dataset = data.SignLanguageKeypoints(
        data_root=data_root,
        images_dir=images_dir,
        labels_dir=labels_dir,
        img_size=training_cfg["img_size"],
        heatmap_size=training_cfg["heatmap_size"],
        sigma=training_cfg["sigma"],
        augmentations=val_aug,
        training=False,
        file_stems=val_stems,
    )
    return train_dataset, val_dataset


def create_optimizer(cfg: Dict[str, Any], model_params) -> torch.optim.Optimizer:
    optim_cfg = cfg.get("optimizer", {})
    name = optim_cfg.get("name", "adam").lower()
    lr = optim_cfg.get("lr", 3e-4)
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    if name == "adam":
        betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(model_params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw":
        betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(model_params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "sgd":
        momentum = optim_cfg.get("momentum", 0.9)
        return torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise KeyError(f"Unsupported optimizer: {name}")


def create_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_cfg = cfg.get("scheduler", {})
    name = sched_cfg.get("name", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    if name == "step":
        step_size = sched_cfg.get("step_size", 10)
        gamma = sched_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise KeyError(f"Unsupported scheduler: {name}")


def evaluate(
    loader: DataLoader,
    model_module: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    model_module.eval()
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_vis: List[torch.Tensor] = []
    all_bboxes: List[torch.Tensor] = []
    total_loss = 0.0
    total_batches = 0
    img_size = cfg["training"]["img_size"]
    heatmap_size = cfg["training"]["heatmap_size"]
    scale = img_size / heatmap_size
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            target_heatmaps = batch["heatmaps"].to(device)
            visibility = batch["visibility"].to(device)
            outputs = model_module(images)
            loss = criterion(outputs, target_heatmaps, visibility)
            total_loss += loss.item()
            total_batches += 1
            coords_heatmap, conf = metrics.decode_heatmaps(outputs.detach())
            coords = coords_heatmap * scale
            all_preds.append(coords.cpu())
            all_targets.append(batch["keypoints"].cpu())
            all_vis.append(visibility.cpu())
            bboxes = []
            for meta in batch["meta"]:
                bbox = meta.get("bbox")
                if bbox is None:
                    bbox = [0.0, 0.0, float(img_size), float(img_size)]
                bboxes.append(bbox)
            all_bboxes.append(torch.tensor(bboxes, dtype=torch.float32))
    if total_batches == 0:
        return {"val_loss": 0.0}
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    visibility = torch.cat(all_vis, dim=0)
    bboxes = torch.cat(all_bboxes, dim=0)
    metrics_dict = metrics.compute_metrics(preds, targets, visibility, bboxes)
    metrics_dict["val_loss"] = total_loss / total_batches
    return metrics_dict


def train() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    cfg = apply_cli_overrides(cfg, args)
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    utils.set_seed(train_cfg.get("seed", 42))

    device_str = train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    LOGGER.info("Using device: %s", device)

    train_dataset, val_dataset = create_datasets(cfg)
    train_loader = data.build_dataloaders(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=True,
    )
    val_loader = data.build_dataloaders(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=False,
    )

    model_module = model.build_model(cfg).to(device)
    criterion = losses.build_loss(cfg).to(device)
    optimizer = create_optimizer(cfg, model_module.parameters())
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler(enabled=train_cfg.get("mixed_precision", False))

    ema_decay = cfg.get("logging", {}).get("ema_decay", 0.0)
    ema = utils.ExponentialMovingAverage(model_module, ema_decay) if ema_decay > 0 else None

    outdir = Path(paths_cfg.get("output_dir", "outputs/default"))
    outdir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(cfg["paths"].get("tensorboard_dir", "runs")) / outdir.name
    tb_logger = utils.TensorBoardLogger.create(tb_dir)

    start_epoch = 0
    best_metric = 0.0
    resume_path = train_cfg.get("resume")
    if resume_path:
        checkpoint = utils.load_checkpoint(Path(resume_path), map_location=device_str)
        model_module.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", scaler.state_dict()))
        start_epoch = checkpoint.get("epoch", 0)
        best_metric = checkpoint.get("best_metric", 0.0)
        LOGGER.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    total_epochs = train_cfg["epochs"]
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, total_epochs):
        model_module.train()
        epoch_loss = 0.0
        start_time = time.time()
        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            target_heatmaps = batch["heatmaps"].to(device)
            visibility = batch["visibility"].to(device)
            with amp.autocast(enabled=train_cfg.get("mixed_precision", False)):
                outputs = model_module(images)
                loss = criterion(outputs, target_heatmaps, visibility) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model_module)
            epoch_loss += loss.item() * grad_accum
            if scheduler and (step + 1) % grad_accum == 0:
                scheduler.step()
            global_step += 1

        duration = time.time() - start_time
        LOGGER.info("Epoch %d/%d - loss: %.4f - time: %.1fs", epoch + 1, total_epochs, epoch_loss / len(train_loader), duration)
        tb_logger.log_scalar("train/loss", epoch_loss / len(train_loader), epoch + 1)

        # Validation
        eval_model = model_module
        if ema:
            ema.apply(model_module)
            eval_model = model_module
        metrics_dict = evaluate(val_loader, eval_model, criterion, device, cfg)
        if ema:
            ema.restore(model_module)
        LOGGER.info("Validation metrics: %s", metrics_dict)
        for key, value in metrics_dict.items():
            tb_logger.log_scalar(f"val/{key}", value, epoch + 1)

        metric_for_selection = metrics_dict.get("pck@0.1", 0.0)
        checkpoint_state = {
            "epoch": epoch + 1,
            "model_state": model_module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_metric": best_metric,
        }
        utils.save_checkpoint(checkpoint_state, outdir / "last.ckpt")
        if metric_for_selection > best_metric:
            best_metric = metric_for_selection
            checkpoint_state["best_metric"] = best_metric
            utils.save_checkpoint(checkpoint_state, outdir / "best.ckpt")
            LOGGER.info("New best model with PCK@0.1 = %.4f", best_metric)

    tb_logger.close()
    LOGGER.info("Training complete. Checkpoints saved to %s", outdir)


if __name__ == "__main__":
    train()
