# Architecture & Engineering Notes

This document complements the README by detailing the internal structure of the `signlang-skeleton` project, including model topology, data processing, losses, metrics, and configuration patterns. The goal is to make it easy to extend or replace components without rewriting the entire pipeline.

## Data Pipeline
- **Input**: 512×512 RGB images produced by Blender alongside per-image JSON annotations.
- **Dataset**: `signlang.data.SignLanguageKeypoints` manages image loading, annotation validation, augmentation, and heatmap generation.
  - Expects mirrored folder structure: `images/` (JPG/PNG) and `labels/` (JSON) with matching stems.
  - Validates that all 64 keypoints are provided, coordinates are finite, and widths/heights match the image.
  - Augmentations (Albumentations) include horizontal flip (with left/right keypoint swapping), rotation ±30°, scaling 0.75–1.25, color jitter, and optional blur/noise. Each augmentation can be toggled from `config.yaml`.
  - Heatmaps: Generated at 128×128 (configurable) using 2D Gaussians (sigma configurable). Invisible keypoints (visible = 0) yield zero heatmaps.

## Model Overview

```
Input (B, 3, 512, 512)
 └─ Backbone (ResNet-18 or MobileNetV3-Small; final feature map stride 32)
     └─ Feature tensor (B, C_backbone, 16, 16)
         └─ Deconvolution/Upsample head (configurable depth)
             └─ Heatmap tensor (B, 64, 128, 128)
```

### Backbones
- **ResNet-18** (`backbone=resnet18`): Torchvision implementation without pretrained weights by default. Modify `model.py` to enable pretrained weights if allowed for your use case.
- **MobileNetV3-Small** (`backbone=mobilenet_v3_small`): Lightweight alternative for realtime inference.
- To add a new backbone, implement a factory function returning `(feature_channels, feature_stride, backbone_module)` and extend the `BACKBONES` dictionary.

### Head (Deconvolution/Upsample)
- Three-stage transposed convolution head (default) that upsamples from 16×16 to 128×128.
- Each block: `ConvTranspose2d` → `BatchNorm2d` → `ReLU`. Kernel size and stride are configurable.
- Final 1×1 convolution produces 64-channel heatmaps.
- Modify the `head_spec` list in `build_head` to adjust the number of layers or channels.

## Losses
- **MSELoss**: Default mean-squared error between predicted and ground-truth heatmaps.
- **Focal MSE**: Optional variant that downweights easy examples; useful when many keypoints are occluded.
- Loss selection occurs in `train.py` via the `build_loss` helper in `losses.py`.

## Metrics
- **PCK@0.05 / PCK@0.1**: Percentage of Correct Keypoints using the reference bounding box from annotations. Implemented in `metrics.py` with configurable thresholds.
- **Per-Keypoint MSE**: MSE in pixel space after decoding peaks.
- **OKS-like Score**: Soft accuracy inspired by COCO’s Object Keypoint Similarity.
- Metrics operate on decoded keypoint coordinates; refer to `signlang.metrics.decode_heatmaps` for peak extraction details.

## Configuration & Overrides
- Defaults reside in `signlang/config.yaml`. Parameters include paths, image/heatmap size, augmentation toggles, optimizer settings, and training hyperparameters.
- CLI arguments in `train.py` and `eval.py` override YAML values.
- Use `--config custom.yaml` to load your own configuration file (which can inherit via YAML anchors).

## Training Loop Structure
1. Build datasets and loaders based on config/CLI.
2. Instantiate the model, loss, optimizer, LR scheduler (optional), and EMA helper (optional).
3. Training epoch:
   - For each batch: forward, loss computation, backward, optimizer step. AMP supported via `torch.cuda.amp.autocast` and `GradScaler`.
   - Log metrics to stdout and TensorBoard (`runs/`).
   - Save checkpoints (`best.ckpt` based on validation PCK@0.1 and `last.ckpt`).
4. Validation occurs at the end of each epoch.

## Extending the Repository
- **Adding augmentations**: Modify `build_augmentations` in `data.py`.
- **Changing keypoints**: Edit `keypoints.py` (list and skeleton). Ensure left/right flip mapping stays correct.
- **Alternate heads**: Implement new modules in `model.py` and swap in `build_model`.
- **Different loss**: Extend `losses.py` with your criterion and update `build_loss` to recognize it.
- **New metrics**: Add to `metrics.py` and include in `evaluate` helper.

## Deployment Notes
- `infer_image.py` provides a ready-to-use script for offline predictions.
- `export_onnx.py` exports to ONNX with dynamic batch dimension for deployment in inference engines.
- `live_demo.py` showcases realtime inference; tune the decoder threshold for responsiveness.

## Style Guide
- Type hints are provided throughout the codebase.
- Functions remain concise with targeted responsibilities.
- Docstrings follow Google-style formatting.
- Logging uses the standard library (`logging`) for consistent output.

Happy prototyping!
