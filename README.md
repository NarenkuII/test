# signlang-skeleton

## Project Overview
`signlang-skeleton` is a production-ready PyTorch skeleton for prototyping a 2D keypoint detector tailored for sign language understanding. The project focuses on learning from synthetic Blender-generated RGB images at 512×512 resolution and predicting 64 keypoints covering full body pose, both hands, and minimal facial anchors. The pipeline follows a heatmap regression approach where each keypoint is represented by a Gaussian heatmap at 128×128 spatial resolution, enabling precise localization and stable training. The repository aims to be lightweight yet extensible, offering clean APIs, configuration-driven experiments, and visualization tools suitable for rapid experimentation.

Why 512×512? The resolution balances hand detail (critical for sign language) with GPU memory cost. With mixed precision training, a single RTX 2080 (8 GB) can comfortably handle batch sizes around 16, while larger GPUs (RTX 3090) benefit from faster convergence.

## Installation
Requires Python 3.10+ and PyTorch 2.x (CPU or CUDA). A fresh virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Layout & Naming Rules
Expected dataset directory layout:

```
dataset/
  images/
    000001.jpg
    000002.jpg
    ...
  labels/
    000001.json
    000002.json
    ...
```

- Image formats: `.jpg` (recommended) and `.png` supported.
- Annotation format: per-image JSON files with the same stem as the image.
- All 64 keypoints must be present in each JSON; if occluded, set `visible: 0` with approximate coordinates when possible.

### JSON Specification
Each JSON annotation must conform to the following schema:

```json
{
  "image": "000123.jpg",
  "width": 512,
  "height": 512,
  "keypoints": [
    {"name": "nose", "x": 255.3, "y": 120.7, "visible": 1},
    {"name": "left_eye", "x": 240.2, "y": 110.9, "visible": 1},
    ...,
    {"name": "right_wrist", "x": 300.0, "y": 320.0, "visible": 1},
    {"name": "left_hand_thumb_tip", "x": 200.1, "y": 350.4, "visible": 0},
    ...
  ],
  "bbox": [x_min, y_min, width, height]
}
```

Keypoint list (ordered as expected by the dataset and model):

1. nose
2. left_eye
3. right_eye
4. left_ear
5. right_ear
6. left_shoulder
7. right_shoulder
8. left_elbow
9. right_elbow
10. left_wrist
11. right_wrist
12. left_hip
13. right_hip
14. left_knee
15. right_knee
16. left_ankle
17. right_ankle
18. left_hand_wrist
19. left_hand_thumb_cmc
20. left_hand_thumb_mcp
21. left_hand_thumb_ip
22. left_hand_thumb_tip
23. left_hand_index_mcp
24. left_hand_index_pip
25. left_hand_index_dip
26. left_hand_index_tip
27. left_hand_middle_mcp
28. left_hand_middle_pip
29. left_hand_middle_dip
30. left_hand_middle_tip
31. left_hand_ring_mcp
32. left_hand_ring_pip
33. left_hand_ring_dip
34. left_hand_ring_tip
35. left_hand_little_mcp
36. left_hand_little_pip
37. left_hand_little_dip
38. left_hand_little_tip
39. right_hand_wrist
40. right_hand_thumb_cmc
41. right_hand_thumb_mcp
42. right_hand_thumb_ip
43. right_hand_thumb_tip
44. right_hand_index_mcp
45. right_hand_index_pip
46. right_hand_index_dip
47. right_hand_index_tip
48. right_hand_middle_mcp
49. right_hand_middle_pip
50. right_hand_middle_dip
51. right_hand_middle_tip
52. right_hand_ring_mcp
53. right_hand_ring_pip
54. right_hand_ring_dip
55. right_hand_ring_tip
56. right_hand_little_mcp
57. right_hand_little_pip
58. right_hand_little_dip
59. right_hand_little_tip
60. face_nose
61. face_left_eye
62. face_right_eye
63. face_left_ear
64. face_right_ear

`visible` must be either 0 (occluded) or 1 (visible). When omitted, the loader defaults to 1. Values outside image bounds raise an error.

## Training
Quick start (uses defaults from `signlang/config.yaml`):

```bash
python -m signlang.train --data-root dataset \
  --images-dir images --labels-dir labels \
  --outdir outputs/run1 --epochs 50 --batch-size 16 \
  --backbone resnet18 --img-size 512 --mixed-precision
```

### Advanced Configuration
- Override core settings via CLI (e.g., `--lr 1e-4`, `--batch-size 8`). Fine-tune `optimizer`, `sigma`, or `heatmap_size` directly in `signlang/config.yaml`.
- Edit `signlang/config.yaml` to change `img_size`, `heatmap_size`, `sigma`, `augmentations`, `optimizer`, `scheduler`, `epochs`, and `num_workers`.
- Modify `model.py` to customize backbone depth (e.g., swap to `mobilenet_v3_small`), widen deconvolution head layers, or alter output stride.
- Mixed precision (`--mixed-precision`) cuts VRAM usage and speeds up training on CUDA GPUs.
- For GPUs with limited memory, reduce `batch_size`, enable gradient accumulation (see comments in `train.py`), or lower `img_size`/`heatmap_size` proportionally.

## Evaluation
Evaluate on the validation split (auto 90/10 split if none provided):

```bash
python -m signlang.eval --data-root dataset --images-dir images --labels-dir labels --checkpoint outputs/run1/best.ckpt
```

Outputs PCK@0.05, PCK@0.1, per-keypoint MSE, and OKS-like metrics.

## Inference (Single Image)

```bash
python -m signlang.infer_image --checkpoint outputs/run1/best.ckpt --image path/to/img.jpg --out out.jpg
```

The script renders the predicted skeleton over the input image and saves the visualization to `out.jpg`. Use `--show` to display the overlay interactively.

## Optional Live Demo

```bash
python -m signlang.live_demo --checkpoint outputs/run1/best.ckpt
```

Displays webcam frames with the inferred skeleton in real time. Specify `--conf-threshold` to filter low-confidence peaks and `--device cpu` to run without CUDA.

## Export to ONNX

```bash
python -m signlang.export_onnx --checkpoint outputs/run1/best.ckpt --output signlang.onnx
```

- Dynamic batch dimension supported.
- Input: `(N, 3, 512, 512)` RGB tensors normalized to `[0, 1]`.
- Output: `(N, 64, 128, 128)` heatmaps.

## Troubleshooting
- **VRAM limits**: Lower `batch_size` or `img_size`, enable `--mixed-precision`.
- **Training diverges**: Check JSON annotations for invalid coordinates or missing keypoints.
- **Performance plateau**: Try `mobilenet_v3_small` for faster inference, adjust `sigma` or `heatmap_size`, or enable/disable augmentations.
- **Resume training**: Use `--resume outputs/run1/last.ckpt`.
- **Logging**: TensorBoard logs are written to `runs/`; launch with `tensorboard --logdir runs`.

## Modifying the Architecture
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a deep dive on the model, loss functions, metrics, and configuration hooks. The head layers in `signlang/model.py` are modular and heavily commented to guide customizations (e.g., widening deconvolution filters, stacking additional upsample blocks, or swapping in a transformer backbone).

## License
Released under the [MIT License](LICENSE).
