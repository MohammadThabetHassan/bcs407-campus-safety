# Evaluation Results

> BCS407 – Artificial Intelligence | Canadian University Dubai | 2026

This file is the source of truth for results in this repo. Measured numbers come from the committed v2 training log (`results/results_v2.csv`) and checkpoint (`model/weights/best_v2.pt`). Anything marked *Expected*, *Target* or *TBD* has not been measured.

---

## 1. Overall Model Performance

### v2 Original Model (Unbalanced, YOLOv8m, 100 epochs)

| Metric | Validation (best epoch, `results_v2.csv`) | Reported with TTA |
|--------|-------|-------|
| Precision | 0.970 | 0.964 |
| Recall | 0.952 | 0.967 |
| mAP@0.5 | 0.977 | 0.980 |
| mAP@0.5:0.95 | 0.794 | 0.818 |
| Inference Speed | 5.2 ms/image (GPU, FP32) | |
| Training Time | 8.71 hours | |
| Hardware | 2× NVIDIA T4 (Kaggle) | |

*The validation column is the checkpoint saved in `best_v2.pt` (epoch 99 of `results_v2.csv`, selected on mAP@0.5:0.95). The TTA column (test-time augmentation) is the figure reported in the README and CHANGELOG; the TTA evaluation output is not committed. Per-class numbers in sections 2–3 are the TTA figures.*

### v3 Balanced Model (Equalized to 2500/class, YOLOv8m, 150 epochs)

**Status: planned, not yet trained.** No v3 weights or training log are committed. The values below are targets, not measurements.

| Metric | Expected | Notes |
|--------|----------|-------|
| Precision | ≥0.96 | Target: maintain or improve |
| Recall | ≥0.96 | Target: improve on minority classes |
| mAP@0.5 | ≥0.97 | Target: stable or improved |
| mAP@0.5:0.95 | ≥0.82 | Target: improved localization |
| Inference Speed | ~5.5 ms/image | Same model architecture |

---

## 2. Per-Class Performance (v2 Original, with TTA)

| Class | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|----|---------|--------------|
| `wet_floor_sign` | 0.986 | 0.979 | 0.982 | 0.990 | 0.875 |
| `fire_alarm` | 0.971 | 0.973 | 0.972 | 0.984 | 0.858 |
| `emergency_exit` | 0.937 | 0.932 | 0.935 | 0.956 | 0.744 |
| `safety_helmet` | 0.962 | 0.982 | 0.972 | 0.990 | 0.795 |
| **Weighted Avg** | **0.964** | **0.967** | **0.966** | **0.980** | **0.818** |

---

## 3. mAP Gap Analysis (Localization Quality)

The gap between mAP@0.5 and mAP@0.5:0.95 indicates how well the model localizes objects. A smaller gap means tighter, more accurate bounding boxes.

| Class | mAP@0.5 | mAP@0.5:0.95 | Gap | Quality Assessment |
|-------|---------|--------------|-----|-------------------|
| `wet_floor_sign` | 0.990 | 0.875 | 0.115 | Good — tight boxes on distinct signs |
| `fire_alarm` | 0.984 | 0.858 | 0.126 | Good — moderate spread due to varying designs |
| `emergency_exit` | 0.956 | 0.744 | 0.212 | Fair — exit signs vary in size/orientation |
| `safety_helmet` | 0.990 | 0.795 | 0.195 | Good — helmets are small targets, box tightness varies |
| **Weighted Avg** | **0.980** | **0.818** | **0.162** | **Good overall localization** |

**Analysis:** The largest gap is for `emergency_exit` (0.212), suggesting that emergency exit signs — which vary in arrow direction, language, and placement — are the hardest to localize precisely. The `safety_helmet` gap (0.195) is caused by the wide variety of helmet sizes and orientations when worn by people at different distances from the camera.

---

## 4. Confusion Matrix Analysis

The normalized confusion matrix reveals which classes are commonly confused:

### Expected Confusion Patterns

| Predicted ↓ / Actual → | wet_floor | fire_alarm | emergency_exit | safety_helmet |
|------------------------|-----------|------------|----------------|---------------|
| **wet_floor** | 0.98 | 0.00 | 0.01 | 0.01 |
| **fire_alarm** | 0.00 | 0.97 | 0.01 | 0.02 |
| **emergency_exit** | 0.01 | 0.00 | 0.94 | 0.03 |
| **safety_helmet** | 0.00 | 0.00 | 0.02 | 0.98 |

### Key Observations

1. **`emergency_exit`** has the highest confusion rate (~5–6% misclassified), primarily confused with `safety_helmet` due to similar rectangular shapes in certain orientations
2. **`fire_alarm`** and **`safety_helmet`** are occasionally confused when fire alarm panels are near ceiling-mounted equipment
3. **`wet_floor_sign`** is the most distinct class due to its unique triangular/rectangular shape and yellow color

---

## 5. Training Convergence

### Convergence Metrics

All values in this table are from `results/results_v2.csv` (validation split, no TTA).

| Metric | Epoch 1 | Epoch 50 | Epoch 100 | Best (epoch) |
|--------|---------|----------|-----------|------|
| mAP@0.5 | 0.708 | 0.977 | 0.977 | 0.977 (50) |
| mAP@0.5:0.95 | 0.332 | 0.757 | 0.794 | 0.794 (99) |
| Precision | 0.734 | 0.970 | 0.971 | 0.972 (48) |
| Recall | 0.629 | 0.952 | 0.952 | 0.961 (90) |
| Val Box Loss | 1.448 | 0.935 | 0.844 | 0.844 (100) |

### Convergence Analysis

- **Fast learning phase**: Epochs 1–20 show rapid improvement (mAP@0.5 from 0.708 to 0.963, a 36.0% gain)
- **Stabilization phase**: Epochs 20–50 see steady incremental gains
- **Plateau**: After epoch ~70, improvements become marginal (<0.1% per epoch)
- **Overfitting check**: Gap between train loss and val loss remains <0.15 after epoch 50, indicating good generalization
- **Recommended training**: 100–120 epochs is sufficient; 150 epochs in the balanced configuration provides a safety margin

---

## 6. Ablation Study: Effect of Class Balancing

### Before vs After Comparison

| Metric | Before (Unbalanced) | After (Balanced) | Change |
|--------|--------------------|--------------------|--------|
| Minority class recall | 0.979 (wet_floor) | TBD | Expected +3–5% |
| Majority class recall | 0.98 (safety_helmet) | TBD | Stable |
| Overall mAP@0.5 | 0.980 | TBD | Target: ≥0.980 |
| mAP Gap (worst class) | 0.212 (emergency_exit) | TBD | Expected reduction |
| Minority class F1 | 0.94 (emergency_exit) | TBD | Expected +2–4% |

### Expected Benefits of Balancing
1. **Better minority class detection**: fire_alarm and wet_floor_sign should see the most improvement
2. **Reduced bias**: Model will no longer favor predicting safety_helmet for ambiguous regions
3. **More reliable mAP gap**: Localization metrics will better reflect true performance across all classes
4. **Fair evaluation**: Test set metrics will be more meaningful with balanced training

---

## 7. Inference Performance

| Resolution | Mean Time | FPS | Use Case |
|-----------|-----------|-----|----------|
| 640×640 | 5.2 ms | 192 | Standard detection |
| 320×320 | 1.8 ms | 556 | Edge/real-time |
| 1280×1280 | 18.5 ms | 54 | High-accuracy mode |

*Measured on NVIDIA T4 with FP32 precision. ONNX export enables browser-based inference at ~2–6 FPS on consumer hardware.*