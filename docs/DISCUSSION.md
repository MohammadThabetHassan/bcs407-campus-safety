# Discussion

> BCS407 – Artificial Intelligence | Canadian University Dubai | 2026

---

## 1. Summary of Results

Our YOLOv8m-based campus safety detection system achieved strong overall performance:

| Metric | Result | Assessment |
|--------|--------|------------|
| mAP@0.5 | 0.980 | Excellent (>0.95 threshold) |
| mAP@0.5:0.95 | 0.818 | Good (tight bounding boxes) |
| Precision | 0.964 | Low false positive rate |
| Recall | 0.967 | Low false negative rate |
| Inference Speed | 5.2 ms/img | Real-time capable (>30 FPS on GPU) |

### Per-Class Breakdown

| Class | mAP@0.5 | mAP@0.5:0.95 | Strengths | Challenges |
|-------|---------|--------------|-----------|------------|
| `wet_floor_sign` | 0.990 | 0.875 | High contrast, distinct shape | Small size in some scenes |
| `fire_alarm` | 0.984 | 0.858 | Consistent red panel design | Variety in mounting heights |
| `emergency_exit` | 0.956 | 0.744 | Good detection rate | Arrow direction variations, low localization |
| `safety_helmet` | 0.990 | 0.795 | Large dataset, strong detection | Scale variation due to distance |

---

## 2. Comparison with Related Work

| Method | Backbone | Classes | mAP@0.5 | mAP@0.5:0.95 | Dataset Size | Deployment |
|--------|----------|---------|---------|--------------|-------------|------------|
| **Ours (v3, balanced)** | **YOLOv8m** | **4** | **0.980** | **0.818** | **10,000** | **GPU/Edge** |
| Fang et al. [4] | Faster R-CNN | 1 (helmet) | 0.92 | — | ~2,000 | Server |
| Wang et al. [5] | YOLOv4 | 3 (PPE) | 0.95 | — | ~5,000 | Jetson Nano |
| Chen et al. [12] | ResNet-50 | 3 (hazard) | 0.89 | — | ~3,000 | Server |
| Almalki et al. [11] | YOLOv5s | 2 (person/vehicle) | 0.91 | — | ~4,000 | Jetson Nano |

**Key advantages of our work:**
- **More classes**: 4 safety categories vs. 1–3 in existing work
- **Class imbalance addressed**: First work to systematically quantify and correct imbalance in campus safety detection
- **Higher mAP**: 0.980 mAP@0.5 exceeds all comparable systems
- **Both indoor and PPE detection**: Combines environmental hazards + safety equipment

---

## 3. Key Findings

### What Worked Well
1. **YOLOv8m as backbone**: Achieved the best accuracy among YOLOv8 variants without excessive compute
2. **Offline augmentation + online augmentation**: Combining both strategies yielded diverse training samples
3. **Class equalization**: Improved minority class recall by an estimated 3–5%
4. **Cosine annealing with warmup**: Smooth convergence without learning rate oscillation

### Surprising Results
1. **Safety helmet performance**: Despite having 5× more training images than wet_floor_sign, mAP@0.5 was comparable (0.990 vs 0.990), suggesting the model was already near saturation for these classes
2. **mAP gap for safety_helmet** (0.195) was nearly as large as for emergency_exit (0.212), despite higher absolute mAP — small helmets at distance have inherently variable bounding boxes
3. **Wet floor signs achieved the best mAP** despite having the fewest training images, likely due to their high visual distinctiveness

### Training Observations
- Validation loss plateaued at epoch ~70 for the original model
- The balanced model required ~80 epochs to reach comparable performance (due to effective doubling of training data)
- Early epochs showed faster improvement for majority classes; minority classes caught up after epoch ~40

---

## 4. Limitations

### Data Limitations
1. **Indoor only**: All datasets were collected from indoor campus environments; outdoor scenarios (parking lots, sports fields) are not covered
2. **Fixed cameras**: All training images appear to be from static, ceiling-mounted cameras; mobile/body-worn camera performance is untested
3. **Lighting conditions**: Dataset predominantly represents well-lit indoor lighting; performance in low-light or emergency lighting is unknown
4. **Occlusion handling**: Partially occluded safety objects (e.g., fire alarm behind a poster) are underrepresented

### Model Limitations
1. **No temporal reasoning**: Current system processes individual frames; cannot detect trends (e.g., missing helmet for >5 minutes)
2. **No severity assessment**: Detects presence/absence but cannot assess severity (e.g., partially visible exit sign vs. completely blocked)
3. **Single-image inference**: No multi-frame fusion or tracking for consistent alerts

### Evaluation Limitations
1. **Single test split**: No k-fold cross-validation; results may vary with different random splits
2. **No domain shift testing**: Training and test sets from same distribution; real-world deployment on different campuses may show degraded performance
3. **No false alarm analysis**: We report precision but do not analyze the real-world cost of false positive alerts (e.g., alert fatigue for security staff)

---

## 5. Future Work

### Short-Term Improvements
1. **Video-based detection**: Extend to temporal analysis using YOLOv8 with ByteTrack for object tracking across frames
2. **Alert system integration**: Connect detection output to automated notification system (email/SMS to facilities management)
3. **Quantify false alarm impact**: Deploy in pilot environment and measure alert accuracy over 30 days
4. **Expand dataset**: Add night-time and low-light images; include more camera angles

### Medium-Term Enhancements
5. **Edge deployment**: Optimize model with TensorRT or ONNX Runtime for NVIDIA Jetson or Raspberry Pi deployment
6. **Multi-camera fusion**: Combine detections from multiple cameras for 3D localization of safety violations
7. **Anomaly detection**: Go beyond classification to detect anomalous events (e.g., fire alarm being covered, exit sign being removed)
8. **Severity scoring**: Add confidence-weighted severity assessment for each detection

### Long-Term Vision
9. **Campus-wide safety dashboard**: Real-time visualization of all safety compliance metrics across campus
10. **Predictive maintenance**: Use detection patterns to predict when safety equipment needs replacement
11. **Multi-modal integration**: Combine visual detection with IoT sensor data (smoke detectors, water leak sensors)
12. **Transfer learning pipeline**: Pre-train on larger datasets (COCO → BDD100K → campus safety) for better generalization

---

## 6. Ethical Reflections

This system demonstrates the power of AI for safety monitoring, but must be deployed responsibly (see [Ethics Analysis](ETHICS.md) for full discussion). Key considerations include:

- **Privacy by design**: No facial recognition, no personal data storage, real-time-only processing
- **Human oversight**: System should augment, not replace, human safety officers
- **Bias monitoring**: Regular audits needed to ensure equitable detection across different lighting, skin tones, and campus areas
- **Transparency**: All detection results should be auditable and explainable