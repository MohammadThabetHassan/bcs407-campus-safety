#!/usr/bin/env bash
# Original v2 training script — kept for reproducibility comparison
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Original v2 Training (for comparison) ==="

yolo detect train \
  data=dataset/data.yaml \
  model=yolov8m.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  workers=0 \
  cache=False \
  cos_lr=True \
  lr0=0.01 \
  lrf=0.001 \
  warmup_epochs=5 \
  project=runs \
  name=campus_safety_v2_original \
  exist_ok=True