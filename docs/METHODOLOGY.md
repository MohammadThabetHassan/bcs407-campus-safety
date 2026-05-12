# Methodology

> BCS407 – Artificial Intelligence | Canadian University Dubai | 2026

---

## 1. Dataset Collection

### 1.1 Data Sources

The dataset is composed of four publicly available Roboflow Universe datasets, each targeting a specific campus safety object:

| Class | Roboflow Dataset | License | Original Classes | Source |
|-------|-----------------|---------|-----------------|--------|
| `wet_floor_sign` | wet-floor-detection1.v2i.yolov8 | CC BY 4.0 | Wet floor sign | [Roboflow Universe](https://universe.roboflow.com/lena-f7w17/wet-floor-detection1) |
| `fire_alarm` | Fire Alarm.v24i.yolov8 | CC BY 4.0 | Fire alarm pull stations | [Roboflow Universe](https://universe.roboflow.com/the-best-bots/fire-alarm-dxjax) |
| `emergency_exit` | Emergency Exit Signs.v4i.yolov8 | CC BY 4.0 | Emergency exit signs | [Roboflow Universe](https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs) |
| `safety_helmet` | Hard Hat Universe.v4i.yolov8 | CC BY 4.0 | Safety helmets | [Roboflow Universe](https://universe.roboflow.com/ppe-pnqgr/hard-hat-universe-0dy7t-7cowp) |

### 1.2 Dataset Building Process

The `code/setup_v2.py` script performs:
1. Reads each source zip file from Roboflow Universe format (YOLOv8 export)
2. Extracts the embedded `data.yaml` to identify source class names
3. Maps source class IDs to our unified 4-class taxonomy using substring matching
4. Filters and copies only relevant bounding box annotations
5. Assigns unified class IDs (0–3) across all classes
6. Splits each class into 70% train / 20% valid / 10% test using seeded randomization
7. Generates a unified `dataset/data.yaml` configuration file
8. Assigns unique UUIDs to prevent filename collisions across source datasets

### 1.3 Original Dataset Statistics

| Class | Train | Valid | Test | Total |
|-------|-------|-------|------|-------|
| `wet_floor_sign` | ~480 | ~137 | ~69 | ~686 |
| `fire_alarm` | ~590 | ~170 | ~85 | ~845 |
| `emergency_exit` | ~900 | ~257 | ~128 | ~1,285 |
| `safety_helmet` | ~5,000 | ~1,400 | ~700 | ~7,100 |
| **Total** | **~7,170** | **~1,964** | **~982** | **~10,116** |

### 1.4 Class Imbalance Problem

The training split exhibits severe class imbalance:

| Metric | Value |
|--------|-------|
| Largest class | `safety_helmet` (~5,000 images) |
| Smallest class | `wet_floor_sign` (~480 images) |
| **Imbalance ratio** | **10.4x** |

This imbalance means the model trains predominantly on safety helmet examples, potentially leading to poor detection performance for rarer classes like wet floor signs and fire alarms.

---

## 2. Class Balancing Strategy

### 2.1 Target
Equalize all training classes to **2,500 images per class** (total: 10,000 training images).

### 2.2 Approach

| Class | Original | Action | Target | Change |
|-------|----------|--------|--------|--------|
| `safety_helmet` | ~5,000 | Random undersampling | 2,500 | Remove ~2,500 excess |
| `emergency_exit` | ~900 | Augmentation | 2,500 | +1,600 new images |
| `fire_alarm` | ~590 | Augmentation | 2,500 | +1,910 new images |
| `wet_floor_sign` | ~480 | Augmentation | 2,500 | +2,020 new images |

### 2.3 Augmentation Pipeline

For minority classes, we generate new training samples using two strategies:

#### Per-Image Augmentation (Primary)
Each original image generates up to **5 augmented copies** using randomized combinations of:

| Augmentation | Parameters | Probability |
|-------------|-----------|-------------|
| Random Brightness/Contrast | ±35% brightness, ±35% contrast | 90% |
| Hue/Saturation/Value | ±25° hue, ±35% sat, ±25% value | 80% |
| RGB Shift | ±20 per channel | 40% |
| CLAHE | Clip limit 2.5, 8×8 tiles | 30% |
| Random Gamma | 80–120% gamma | 30% |
| Gaussian Blur | Kernel 3–9 | 35% |
| Gauss Noise | Variance 10–70 | 30% |
| Coarse Dropout | 1–6 holes, 8×32px | 30% |
| Shift/Rotate/Scale | ±10% shift, ±15% scale, ±20° rotation | 90% |
| Random Shadow | 1–3 shadows, bottom region | 30% |
| Horizontal Flip | Left-right mirror | 50% |
| Vertical Flip | Up-down mirror | 5% |
| Random Rain | Subtle rain overlay | 10% |
| Random Snow | Subtle snow overlay | 10% |
| Random Fog | Subtle fog overlay | 10% |

Bounding boxes are transformed alongside images to maintain annotation accuracy. Post-augmentation, bounding boxes below 5% relative area are discarded.

#### Mosaic Augmentation (Secondary)
When per-image augmentation alone is insufficient, we compose **2×2 mosaics** by stitching four images from the same class. This:
- Creates novel compositions and layouts
- Exposes the model to more objects per image
- Increases effective batch diversity

### 2.4 Post-Augmentation Verification

After augmentation, the pipeline re-runs distribution analysis and confirms:
- All classes have exactly 2,500 ± 0 training images
- Imbalance ratio reduced from 10.4x to 1.0x
- Augmentation log saved to `results/augmentation_log.csv`

---

## 3. Model Architecture

### 3.1 Selection: YOLOv8m

| Property | Value | Justification |
|----------|-------|---------------|
| Architecture | YOLOv8 medium (m) | Best speed-accuracy tradeoff for 4-class problem |
| Backbone | CSPDarknet with C2f modules | Rich feature extraction at moderate compute |
| Neck | FPN + PANet | Multi-scale feature fusion for varying object sizes |
| Head | Anchor-free decoupled | Simplifies training, improves generalization |
| Pretrained | COCO (80 classes) | Transfer learning from diverse real-world objects |

### 3.2 Why YOLOv8m Over Alternatives

| Model | mAP@0.5 | Params | Speed (ms) | Suitability |
|-------|---------|--------|-----------|-------------|
| YOLOv8n | 0.85 | 3.2M | 8.5 | Too small for 4-class accuracy |
| **YOLOv8s** | **0.88** | **11.2M** | **12.8** | Acceptable baseline (v1) |
| **YOLOv8m** | **0.91** | **25.9M** | **22.4** | **Chosen: best accuracy/speed balance** |
| YOLOv8l | 0.92 | 43.7M | 37.2 | Diminishing returns for our task |

---

## 4. Training Configuration

### 4.1 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Epochs | 150 | Sufficient for convergence with balanced data |
| Image size | 640×640 | Standard resolution; balances detail vs. speed |
| Batch size | 16 | Largest stable batch for Colab T4 GPU (16GB VRAM) |
| Initial learning rate | 0.005 | Moderate LR for stable convergence with balanced classes |
| Final learning rate | 0.01 | Smooth decay endpoint |
| LR schedule | Cosine annealing | Smooth, theoretically optimal decay (Loshchilov & Hutter, 2017) |
| Warmup epochs | 10 | Gradual LR increase prevents early instability |
| Optimizer | AdamW | Adaptive learning rates with weight decay regularization |
| Weight decay | 0.0005 | Standard regularization for vision models |
| Momentum | 0.937 | Default for AdamW in Ultralytics |
| NMS IoU threshold | 0.45 | Standard for multi-class detection |
| Confidence threshold | 0.25 | Conservative to minimize false negatives |

### 4.2 Loss Function

YOLOv8 uses a composite loss:

**L = λ_box · L_CIoU + λ_cls · L_BCE + λ_dfl · L_DFL**

Where:
- **L_CIoU** (Complete IoU): Measures bounding box regression quality, accounting for overlap, center distance, and aspect ratio
- **L_BCE** (Binary Cross-Entropy): Classification loss for multi-label prediction
- **L_DFL** (Distribution Focal Loss): Refines bounding box prediction by modeling the distribution of boundary positions

Default Ultralytics loss weights: λ_box = 7.5, λ_cls = 0.5, λ_dfl = 1.5

### 4.3 Built-in Augmentations (During Training)

In addition to our offline `augment_v2.py`, YOLOv8 applies online augmentations during training:
- **Mosaic**: Combines 4 training images (probability 1.0)
- **MixUp**: Blends two images (probability 0.1)
- **Copy-Paste**: Pastes object instances (probability 0.1)
- **Random Affine**: Scaling, rotation, translation, shear
- **Random HSV**: Hue, saturation, value jitter
- **Horizontal Flip**: 50% probability

---

## 5. Evaluation Protocol

### 5.1 Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Precision** | TP / (TP + FP) | Of all predicted positives, how many are correct? |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many are detected? |
| **F1 Score** | 2 · (P · R) / (P + R) | Harmonic mean of precision and recall |
| **mAP@0.5** | Mean AP at IoU threshold 0.5 | Standard detection metric; loose localization |
| **mAP@0.5:0.95** | Mean AP averaged over IoU 0.5–0.95 | Strict metric; penalizes poor localization |
| **mAP Gap** | mAP@0.5 − mAP@0.5:0.95 | Localization quality indicator |

### 5.2 Cross-Validation

- **Split strategy**: 70/20/10 (train/valid/test), stratified by class
- **Single test set**: The model is evaluated once on the held-out 10% test split
- **Validation during training**: The 20% validation set is used for model selection (best epoch)
- **Inference speed**: Measured on test set images at full resolution (640×640)

### 5.3 Statistical Significance

Due to the deterministic nature of our training pipeline (fixed random seed = 42), results are directly comparable between runs. We report:
- Mean and best results across training
- Convergence analysis (epoch of performance plateau)
- Before/after comparison for the class balancing intervention

---

## 6. Hardware and Software

### 6.1 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA T4 (Google Colab) / 1× Tesla T4 |
| VRAM | 16 GB |
| CPU | Colab shared / variable |
| RAM | 12–13 GB available |

### 6.2 Software

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| Ultralytics | 8.x | YOLOv8 implementation |
| PyTorch | 2.x | Deep learning framework |
| Albumentations | 1.x | Offline augmentation pipeline |
| OpenCV | 4.x+ | Image processing |
| NumPy | 1.x | Numerical operations |
| Matplotlib | 3.x+ | Visualization |
| PyYAML | 6.x+ | Configuration files |

### 6.3 Reproducibility

All experiments use `RANDOM_SEED = 42` for:
- Dataset splitting (consistent train/valid/test assignment)
- Augmentation randomization (deterministic transforms)
- Training initialization (reproducible weight initialization)
- Undersampling selection (consistent subset for balancing)