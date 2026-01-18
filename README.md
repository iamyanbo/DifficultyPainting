# DifficultyPainting

Painting learned detection difficulty onto LiDAR point clouds to improve 3D object detection.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Results (KITTI Validation, 3D AP @ Standard IoU)

### PointPillars + Difficulty

| Class | Easy | Moderate | Hard | IoU |
|-------|------|----------|------|-----|
| Car | 89.92 | 86.53 | 79.47 | 0.70 |
| Pedestrian | 71.97 | 68.34 | 64.61 | 0.50 |
| Cyclist | 89.72 | 85.21 | 80.08 | 0.50 |

**mAP (Moderate): 80.03%**

### SECOND + Difficulty

| Class | Easy | Moderate | Hard | IoU |
|-------|------|----------|------|-----|
| Car | 90.32 | 88.33 | 79.96 | 0.70 |
| Pedestrian | 74.98 | 73.71 | 71.82 | 0.50 |
| Cyclist | 88.65 | 85.49 | 79.08 | 0.50 |

**mAP (Moderate): 82.51%**

## Overview

This project learns detection difficulty from 2D image patches and paints this information onto LiDAR point clouds. Inspired by PointPainting, we add a learned difficulty feature as an additional channel to each LiDAR point.

![Difficulty Overlay](assets/difficulty_overlay_000015.png)
*Detected objects colored by predicted difficulty. Blue = easy, warmer colors = harder.*

## Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 3D Detector │────▶│ Extract IoU/Diff │────▶│ Train Predictor │────▶│ Paint LiDAR     │
│ (baseline)  │     │ Labels           │     │ (ResNet-18)     │     │ (5 channels)    │
└─────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘
                                                                              │
                                                                              ▼
                                                                     ┌─────────────────┐
                                                                     │ Train Detector  │
                                                                     │ (difficulty-    │
                                                                     │  weighted loss) │
                                                                     └─────────────────┘
```

## Installation

```bash
git clone https://github.com/iamyanbo/DifficultyPainting.git
cd DifficultyPainting
pip install -r requirements.txt
```

This project requires [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Follow their installation instructions.

## Usage

### 1. Extract difficulty labels from baseline detector

```bash
python scripts/extract_detection_difficulty.py \
    --result_pkl path/to/baseline_result.pkl \
    --label_dir data/kitti/training/label_2 \
    --image_dir data/kitti/training/image_2 \
    --output difficulty_labels.csv
```

### 2. Train difficulty predictor

```bash
python scripts/train_difficulty_predictor_final.py \
    --csv_path difficulty_labels.csv \
    --image_dir data/kitti/training/image_2 \
    --epochs 50
```

### 3. Generate difficulty maps

```bash
python scripts/generate_difficulty_maps.py \
    --model_path difficulty_predictor_final.pth \
    --image_dir data/kitti/training/image_2 \
    --output_dir data/kitti/training/difficulty_maps
```

### 4. Paint LiDAR point clouds

```bash
python scripts/paint_difficulty.py \
    --lidar_dir data/kitti/training/velodyne \
    --calib_dir data/kitti/training/calib \
    --map_dir data/kitti/training/difficulty_maps \
    --output_dir data/kitti/training/velodyne_painted_difficulty
```

### 5. Train 3D detector with difficulty features

```bash
cd OpenPCDet/tools
python train.py --cfg_file cfgs/kitti_models/pointpillar_difficulty_aware.yaml \
    --batch_size 4 --epochs 80
```

## Difficulty Predictor

The predictor uses ResNet-18 with auxiliary features (depth, bbox dimensions) to predict detection difficulty:

| Configuration | Correlation |
|---------------|-------------|
| Image only | 64.3% |
| Image + Depth | 78.2% |
| Image + Depth + BBox | **89.0%** |

## License

MIT License

## Acknowledgements

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [PointPainting](https://arxiv.org/abs/1911.10150)
