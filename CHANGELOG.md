# Changelog

All notable changes to the BCS407 Campus Safety Detection project.

## [2.0.0] — 2026-03-31

### Changed
- **Model upgrade**: YOLOv8s (v1) → YOLOv8m (v2)
- **Class update**: replaced legacy class with `safety_helmet`
- **Dataset**: rebuilt with 70/20/10 stratified split (was ~88/9/3)
- **Training**: 100 epochs with cosine LR schedule (was 50, fixed LR)
- **Augmentation**: added offline albumentations pipeline (brightness, contrast, HSV, blur, noise, shift/scale/rotate, shadow, H-flip)
- **Dataset size**: ~10,000+ images across 4 classes (was 6,079)
- **GitHub Pages**: migrated workflow from `peaceiris/actions-gh-pages@v3` to `actions/deploy-pages@v4`
- **README**: full rewrite with v2 classes, version history, dataset table, project structure
- **requirements.txt**: added albumentations, roboflow, updated version ranges
- **Live demo**: added canvas bounding box overlay, FPS counter, class color legend, model info card

### Added
- `code/setup_v2.py` — v2 dataset rebuild from zip files
- `code/augment_v2.py` — offline augmentation pipeline with albumentations
- `contributors/CONTRIBUTORS.md` — team roles and contact info
- `LICENSE` — MIT license
- `CHANGELOG.md` — this file
- Model info card on live demo page

### Verified
- Final v2 metrics recorded in README: Precision 0.964, Recall 0.967, mAP@0.5 0.980 (TTA), mAP@0.5:0.95 0.818 (TTA)
- Final v2 per-class metrics documented for all 4 classes
- `code/inference.py` default weights switched to `model/weights/best_v2.pt`
- `make verify` target added for end-to-end local sanity checks using `best_v2.pt`

### Fixed
- Class order in `docs/index.html` JS array (was mismatched with `data.yaml`)
- Broken glob patterns in `augment_v2.py` final stats
- Hardcoded absolute paths in `setup_v2.py` and `augment_v2.py` (now relative)
- `inference.py`: added error handling, input validation, `--weights` flag help text
- `.gitignore`: removed reference to non-existent `merge_campus_safety.py`

### Removed
- legacy class from v1 (replaced by `safety_helmet`)
- `merge_campus_safety.py` reference (v1 script, not in repo)
- Unused TensorFlow.js CDN from live demo

## [1.0.0] — 2026-03-28

### Initial release
- YOLOv8s baseline trained on an older 4-class set
- mAP@0.5 = 0.971, mAP@0.5:0.95 = 0.810
- 6,079 images, ~88/9/3 split, 50 epochs
- GitHub Pages live demo with webcam detection
- Inference script for image/folder/webcam input
