# BCS407 — AI-Based Smart Campus Safety Detection System

**Course:** BCS407 – Artificial Intelligence  
**Institution:** Canadian University Dubai  
**Theme:** Campus Safety Monitoring  
**Model:** YOLOv8s (fine-tuned)  
**Date:** March 2026

## 👥 Team Members / Contributors

- [Mohammad Thabet Hassan](https://github.com/MohammadThabetHassan)
- [Ahmed Sami Alameri](https://github.com/AhmedSamiAlameri)
- [Fahad](https://github.com/fahad6789123)
- [Omar Alraas](https://github.com/omaralraas)
- [Obadah Loul](https://github.com/obadah-loul)

---

## 🎯 Project Overview

This project implements a real-time computer vision object detection system for campus safety monitoring using YOLOv8s. The system detects four critical safety objects found in indoor campus environments.

**Detected Classes:**
| ID | Class | Description |
|----|-------|-------------|
| 0 | `fire_extinguisher` | Fire extinguishers mounted on walls |
| 1 | `emergency_exit` | Emergency exit signs (all arrow directions) |
| 2 | `fire_alarm` | Fire alarm pull stations and devices |
| 3 | `wet_floor_sign` | Wet floor caution signs |

---

## 📊 Results

### Overall Performance (Validation Set)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.971** |
| **mAP@0.5:0.95** | **0.810** |
| **Precision** | **0.935** |
| **Recall** | **0.958** |
| Training Epochs | 50 |
| Training Time | 1.703 hours |
| Inference Speed | ~4 ms/image (NVIDIA L4) |
| Model Size | 22.5 MB |

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| fire_extinguisher | 0.926 | 0.884 | 0.934 | 0.811 |
| emergency_exit | 0.900 | 0.950 | 0.961 | 0.653 |
| fire_alarm | 0.923 | 1.000 | 0.995 | 0.862 |
| wet_floor_sign | 0.991 | 1.000 | 0.995 | 0.915 |

---

## 📁 Dataset

**Total Images:** 6,079 across 4 classes  
**Annotation Tool:** Roboflow Universe  
**Format:** YOLOv8 (YOLO bounding box format)  
**Split:** ~70% train / ~20% valid / ~10% test  

| Class | Train | Valid | Test | Total |
|-------|-------|-------|------|-------|
| fire_extinguisher | 2,934 | 328 | 0 | 3,262 |
| emergency_exit | 1,151 | 91 | 42 | 1,284 |
| fire_alarm | 822 | 11 | 13 | 846 |
| wet_floor_sign | 435 | 107 | 145 | 687 |

**Dataset Sources (Roboflow Universe):**
- Fire Extinguisher: `universe.roboflow.com/fire-extinguisher/fireextinguisher-z5atr`
- Emergency Exit Signs: `universe.roboflow.com/emergency-exit-signs/emergency-exit-signs`
- Fire Alarm: `universe.roboflow.com/the-best-bots/fire-alarm-dxjax`
- Wet Floor Sign: `universe.roboflow.com/lena-f7w17/wet-floor-detection1`

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/bcs407-campus-safety.git
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

### Reproduce Dataset Merge
```bash
# Place 4 zip files in the folder, then:
python code/merge_campus_safety.py
```

### Reproduce Training
```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8s.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=0 \
  cache=False
```

---

## 📂 Project Structure

```
BCS407_Campus_Safety_Project/
├── model/
│   └── weights/
│       └── best.pt              ← trained model (22.5 MB)
├── dataset/
│   └── data.yaml                ← dataset config
├── code/
│   ├── merge_campus_safety.py   ← dataset merge script
│   └── inference.py             ← live inference script
├── results/
│   ├── plots/                   ← confusion matrix, PR curve, F1 curve
│   ├── predictions/             ← sample annotated test images
│   └── results.csv              ← epoch-by-epoch training log
├── report/                      ← project report (add your PDF/docx here)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8s |
| Pretrained | COCO (ImageNet backbone) |
| Epochs | 50 |
| Image Size | 640×640 |
| Batch Size | 16 |
| Optimizer | AdamW (auto) |
| Learning Rate | 0.00125 |
| GPU | NVIDIA L4 (22GB) |
| Augmentations | HSV, horizontal flip, mosaic, mixup |

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
