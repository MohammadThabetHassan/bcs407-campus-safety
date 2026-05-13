#!/usr/bin/env bash
# === Run this cell in Kaggle AFTER the pipeline finishes ===
# Commits all results + trained weights to GitHub

cd /kaggle/working/bcs407-campus-safety

git config user.name "BCS407 Pipeline"
git config user.email "pipeline@bcs407.local"

# Add all results, weights, and figures
git add results/ runs/ dataset/data.yaml code/
git status

git commit -m "feat: v3 balanced training results + figures

- Dataset equalized to 2500 images/class (17.1x -> 1.0x imbalance)
- YOLOv8m trained for 150 epochs on T4 GPU
- Per-class evaluation metrics and confusion matrix
- 14 publication-quality report figures (PNG+PDF)"

git push origin main
