#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Installing ultralytics ==="
pip install ultralytics -q

echo "=== Balanced Training Config (v3) ==="
echo "Model:       YOLOv8m"
echo "Epochs:      150 (increased from 100)"
echo "LR0:         0.005 (reduced from 0.01 for stable convergence)"
echo "LRF:         0.01  (increased from 0.001 for smoother ending)"
echo "Warmup:      10 epochs (increased from 5)"
echo "Batch:       16"
echo "Imgsz:       640"
echo "Class balance: equalized to 2500 per class"
echo ""

python -m ultralytics detect train \
  data=dataset/data.yaml \
  model=yolov8m.pt \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=0 \
  cache=False \
  cos_lr=True \
  lr0=0.005 \
  lrf=0.01 \
  warmup_epochs=10 \
  momentum=0.937 \
  weight_decay=0.0005 \
  project=runs \
  name=campus_safety_v3_balanced \
  exist_ok=True