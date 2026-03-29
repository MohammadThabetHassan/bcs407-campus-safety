#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

yolo detect train \
  data=dataset/data.yaml \
  model=yolov8m.pt \
  epochs=100 \
  imgsz=640 \
  batch=32 \
  device=0 \
  workers=0 \
  cache=False \
  cos_lr=True \
  lr0=0.01 \
  lrf=0.001 \
  warmup_epochs=5 \
  project=runs \
  name=campus_safety_v2_fixed \
  exist_ok=True
