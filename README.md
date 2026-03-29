# BCS407 — AI-Based Smart Campus Safety Detection System

**Course:** BCS407 – Artificial Intelligence
**Institution:** Canadian University Dubai
**Theme:** Campus Safety Monitoring
**Model:** YOLOv8m (v2, fine-tuned) / YOLOv8s (v1, baseline)
**Date:** March 2026

---

## 👥 Team Members / Contributors

| Member | GitHub |
|--------|--------|
| Mohammad Thabet Hassan | [@MohammadThabetHassan](https://github.com/MohammadThabetHassan) |
| Ahmed Sami Alameri | [@AhmedSamiAlameri](https://github.com/AhmedSamiAlameri) |
| Fahad Al Jazzeri | [@fahadALjazzeri](https://github.com/fahadALjazzeri) |
| Omar Alraas | [@omaralraas](https://github.com/omaralraas) |
| Obadah Loul | [@obadah-loul](https://github.com/obadah-loul) |

---

## 🎯 Project Overview

Real-time computer vision object detection system for campus safety monitoring using YOLOv8.
The system detects four critical safety objects found in indoor campus environments.

**Detected Classes:**

| ID | Class | Description |
|----|-------|-------------|
| 0 | `wet_floor_sign` | Wet floor caution signs |
| 1 | `fire_alarm` | Fire alarm pull stations and devices |
| 2 | `emergency_exit` | Emergency exit signs (all arrow directions) |
| 3 | `safety_helmet` | Safety / hard hats worn by personnel |

---

## 📊 Results

### Version History

| Version | Model | Classes | Split | Epochs | mAP@0.5 | mAP@0.5:0.95 | Notes |
|---------|-------|---------|-------|--------|---------|--------------|-------|
| v1 | YOLOv8s | fire_extinguisher, emergency_exit, fire_alarm, wet_floor_sign | ~88/9/3 | 50 | 0.971 | 0.810 | Baseline (old classes) |
| v2 | YOLOv8m | wet_floor_sign, fire_alarm, emergency_exit, safety_helmet | 70/20/10 | 100 | *in progress* | *in progress* | New classes, offline augmentation, cosine LR, TTA eval |

### v1 Baseline — Per-Class Performance (old classes, for reference)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| fire\_extinguisher *(removed)* | 0.926 | 0.884 | 0.934 | 0.811 |
| emergency\_exit | 0.900 | 0.950 | 0.961 | 0.653 |
| fire\_alarm | 0.923 | 1.000 | 0.995 | 0.862 |
| wet\_floor\_sign | 0.991 | 1.000 | 0.995 | 0.915 |

> v2 results will be added here after training completes.

---

## 📁 Dataset

### v2 Dataset (current)

**Total Images:** ~10,000+ across 4 classes
**Annotation Tool:** Roboflow Universe
**Format:** YOLOv8 (YOLO bounding box format)
**Split:** 70% train / 20% valid / 10% test

| Class | Train | Valid | Test | Source |
|-------|-------|-------|------|--------|
| wet\_floor\_sign | ~480+ | ~137+ | ~69+ | [Roboflow](https://universe.roboflow.com/lena-f7w17/wet-floor-detection1) |
| fire\_alarm | ~590+ | ~170+ | ~85+ | [Roboflow](https://universe.roboflow.com/the-best-bots/fire-alarm-dxjax) |
| emergency\_exit | ~900+ | ~257+ | ~128+ | [Roboflow](https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs) |
| safety\_helmet | ~5000+ | ~1400+ | ~700+ | [Roboflow](https://universe.roboflow.com/ppe-pnqgr/hard-hat-universe-0dy7t-7cowp) |

### v1 Dataset (baseline, archived)

**Total Images:** 6,079 — split was ~88/9/3 (imbalanced)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/MohammadThabetHassan/bcs407-campus-safety.git
cd bcs407-campus-safety
pip install -r requirements.txt
```

### Run Inference on an Image

```bash
python code/inference.py --source path/to/image.jpg
```

### Run Inference on a Folder

```bash
python code/inference.py --source path/to/folder/
```

### Reproduce Dataset Build (v2)

```bash
# Place 4 zip files in ~/bcs407_v2/, then:
python code/setup_v2.py
python code/augment_v2.py
```

### Reproduce Training (v2)

```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8m.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=0 \
  cache=False \
  cos_lr=True
```

---

## 📂 Project Structure

```
bcs407-campus-safety/
├── model/
│   └── weights/
│       ├── best.pt           ← v1 trained model (YOLOv8s, 22.5 MB)
│       └── best_v2.pt        ← v2 trained model (YOLOv8m) [coming soon]
├── dataset/
│   └── data.yaml             ← dataset config (v2 classes)
├── code/
│   ├── merge_campus_safety.py ← v1 dataset merge script
│   ├── setup_v2.py            ← v2 dataset rebuild script
│   ├── augment_v2.py          ← offline augmentation pipeline
│   └── inference.py           ← live inference script
├── results/
│   ├── plots/
│   │   ├── v1/               ← v1 confusion matrix, PR curve, F1 curve
│   │   └── v2/               ← v2 results [coming soon]
│   └── results.csv           ← epoch-by-epoch training log
├── docs/
│   └── index.html            ← GitHub Pages live demo
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Training Configuration (v2)

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8m |
| Pretrained | COCO (ImageNet backbone) |
| Epochs | 100 |
| Image Size | 640×640 |
| Batch Size | 16 |
| Optimizer | AdamW (auto) |
| LR Schedule | Cosine (lr0=0.01, lrf=0.001) |
| Warmup Epochs | 5 |
| Augmentations | HSV, flip, mosaic, mixup, copy-paste + offline albumentations |
| GPU | NVIDIA L4 (23GB) |

---

## 🌐 Live Demo

👉 [https://mohammadthabethassan.github.io/bcs407-campus-safety/](https://mohammadthabethassan.github.io/bcs407-campus-safety/)

Uses your webcam to detect safety objects in real time via the browser.

---

## 🔒 Ethics

- No identifiable human faces in any dataset image
- No license plates or personal data
- All datasets sourced under CC BY 4.0 open licenses
- System intended for safety monitoring only — not surveillance

---

## 📚 CLO Alignment

- **CLO-4:** Applied YOLOv8 object detection to solve a real-world campus safety problem
- **CLO-5:** Technical report and presentation demonstrating team collaboration and communication

---

*BCS407 – Artificial Intelligence | Canadian University Dubai | 2026*
