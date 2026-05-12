#!/usr/bin/env python3
"""
Enhanced augmentation pipeline with class balancing support.

Usage:
    python code/augment_v2.py
    python code/augment_v2.py --balance-mode equalize --target-count 2500
    python code/augment_v2.py --balance-mode visualize-only
"""

import argparse
import csv
import os
import random
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import albumentations as A
    import cv2
    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    print("Warning: albumentations or cv2 not available — augmentation skipped")

RANDOM_SEED = 42
CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]
IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".webp")


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Augment + balance the v2 training split.")
    parser.add_argument(
        "--dataset-dir",
        default=str(repo_root / "dataset"),
        help="Dataset directory containing train/valid/test folders.",
    )
    parser.add_argument(
        "--default-ratio",
        type=float,
        default=0.6,
        help="Fraction of each non-minority class to augment (when balance-mode=none).",
    )
    parser.add_argument(
        "--minority-ratio",
        type=float,
        default=1.0,
        help="Fraction of minority-class images to augment (when balance-mode=none).",
    )
    parser.add_argument(
        "--minority-classes",
        nargs="*",
        default=["fire_alarm", "wet_floor_sign"],
        help="Classes treated as minority when balance-mode=none.",
    )
    parser.add_argument(
        "--balance-mode",
        choices=["none", "equalize", "visualize-only"],
        default="none",
        help="none: original behavior | equalize: balance all classes to target",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=2500,
        help="Target number of training images per class when balance-mode=equalize.",
    )
    parser.add_argument(
        "--max-augment-per-image",
        type=int,
        default=5,
        help="Max augmented copies generated from a single original image.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


# ─────────────────────────── augment transforms ───────────────────────────
def build_strong_transform():
    """Rich augmentation pipeline for generating diverse copies."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.9),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.8),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.4),
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=(3, 9), p=0.35),
            A.GaussNoise(var_limit=(10.0, 70.0), p=0.3),
            A.CoarseDropout(max_holes=6, max_height=32, max_width=32,
                            min_holes=1, min_height=8, min_width=8, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20,
                               border_mode=0, value=0, mask_value=None, p=0.9),
            A.RandomShadow(shadow_roi=(0, 0.4, 1, 1), num_shadows_limit=(1, 3),
                           shadow_dimension=5, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_visibility=0.25,
            label_fields=["class_labels"],
        ),
    )


def build_mild_transform():
    """Lighter augmentation for mosaic composites."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=10,
                               border_mode=0, p=0.8),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.25,
                                  label_fields=["class_labels"]),
    )


# ─────────────────────── YOLO ↔ Albumentations helpers ───────────────────
def load_yolo_bboxes(label_path):
    bboxes = []
    if not label_path.exists():
        return bboxes
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if len(coords) == 4:
                    bboxes.append([cls, *coords])
                elif len(coords) >= 8:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h = x2 - x1, y2 - y1
                    bboxes.append([cls, cx, cy, w, h])
            except (ValueError, IndexError):
                continue
    return bboxes


def save_yolo_bboxes(label_path, bboxes):
    with label_path.open("w", encoding="utf-8") as f:
        for bbox in bboxes:
            f.write(f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")


def yolo_to_alb(bboxes, img_w, img_h):
    boxes, classes = [], []
    for b in bboxes:
        cls, cx, cy, w, h = b
        boxes.append([(cx - w/2)*img_w, (cy - h/2)*img_h, (cx + w/2)*img_w, (cy + h/2)*img_h])
        classes.append(int(cls))
    return boxes, classes


def alb_to_yolo(boxes, classes, img_w, img_h):
    bboxes = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.005, min(1.0, w))
        h = max(0.005, min(1.0, h))
        if cx - w/2 < 0:
            w = cx * 2
        if cx + w/2 > 1:
            w = (1 - cx) * 2
        if cy - h/2 < 0:
            h = cy * 2
        if cy + h/2 > 1:
            h = (1 - cy) * 2
        if w > 0.005 and h > 0.005:
            bboxes.append([cls, cx, cy, w, h])
    return bboxes


def resolve_image_path(images_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def apply_augmentation(image_path, bboxes, transform):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    if image.size == 0 or image.shape[0] < 8 or image.shape[1] < 8:
        return None, None
    height, width = image.shape[:2]
    alb_boxes, alb_classes = yolo_to_alb(bboxes, width, height)
    if not alb_boxes:
        return None, None
    try:
        result = transform(image=image, bboxes=alb_boxes, class_labels=alb_classes)
        aug_bboxes = alb_to_yolo(result["bboxes"], result["class_labels"], width, height)
        return result["image"], aug_bboxes
    except Exception:
        return None, None


def build_mosaic(image_paths, bboxes_list, transform, target_w=640, target_h=640):
    """Create a 2x2 mosaic from up to 4 images + their bboxes."""
    import cv2
    images = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None or img.size == 0:
            return None, None
        if img.shape[0] < 4 or img.shape[1] < 4:
            img = cv2.resize(img, (target_w // 2, target_h // 2),
                             interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (target_w // 2, target_h // 2))
        images.append(img)

    if len(images) < 4:
        black = np.zeros((target_h // 2, target_w // 2, 3), dtype=np.uint8)
        while len(images) < 4:
            images.append(black.copy())

    top = np.hstack([images[0], images[1]])
    bottom = np.hstack([images[2], images[3]])
    mosaic = np.vstack([top, bottom])

    all_bboxes = []
    offsets = [(0, 0), (target_w // 2, 0), (0, target_h // 2), (target_w // 2, target_h // 2)]
    for idx, (bx_list, (ox, oy)) in enumerate(zip(bboxes_list, offsets)):
        for cls, cx, cy, w, h in bx_list:
            px_cx = (cx * (target_w // 2)) + ox
            px_cy = (cy * (target_h // 2)) + oy
            px_w = w * (target_w // 2)
            px_h = h * (target_h // 2)
            all_bboxes.append([cls, px_cx / target_w, px_cy / target_h,
                               px_w / target_w, px_h / target_h])

    alb_boxes, alb_classes = yolo_to_alb(all_bboxes, target_w, target_h)
    if not alb_boxes:
        return mosaic, []
    try:
        result = transform(image=mosaic, bboxes=alb_boxes, class_labels=alb_classes)
        aug_bboxes = alb_to_yolo(result["bboxes"], result["class_labels"], target_w, target_h)
        return result["image"], aug_bboxes
    except Exception:
        return mosaic, all_bboxes  # Return original mosaic if transform fails


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"

    if not train_img_dir.exists() or not train_lbl_dir.exists():
        raise FileNotFoundError(f"Training split not found under {dataset_dir}")

    if not HAS_ALB:
        print("albumentations/cv2 not available. Install: pip install albumentations opencv-python-headless")
        return

    # ── Step 1: Load original training labels ──
    class_labels_orig = defaultdict(list)
    for label_file in sorted(train_lbl_dir.glob("*.txt")):
        if label_file.stem.startswith("aug_"):
            continue
        for cls_name in CLASS_NAMES:
            if label_file.stem.startswith(cls_name):
                class_labels_orig[cls_name].append(label_file)
                break

    print("\n" + "=" * 60)
    print("  ORIGINAL TRAINING DATA DISTRIBUTION")
    print("=" * 60)
    for cls in CLASS_NAMES:
        print(f"  {cls:<20}  {len(class_labels_orig[cls]):>5} images")

    total_orig = sum(len(class_labels_orig[c]) for c in CLASS_NAMES)
    print(f"\n  Total original images: {total_orig}")

    # ── Visualize-only mode ──
    if args.balance_mode == "visualize-only":
        print("\n  [visualize-only] No changes made.")
        # Show imbalance ratio
        counts = [len(class_labels_orig[c]) for c in CLASS_NAMES]
        non_zero = [c for c in counts if c > 0]
        if non_zero:
            print(f"  Imbalance ratio: {max(counts)/min(non_zero):.1f}x")
        return

    # ── Equalize mode ──
    if args.balance_mode == "equalize":
        target = args.target_count
        max_per_orig = args.max_augment_per_image

        print(f"\n{'='*60}")
        print(f"  BALANCE MODE: EQUALIZE to {target} per class")
        print(f"{'='*60}")

        plan = {}
        for cls in CLASS_NAMES:
            orig_count = len(class_labels_orig[cls])
            if orig_count >= target:
                plan[cls] = {
                    "action": "undersample",
                    "orig": orig_count,
                    "target": target,
                    "remove": orig_count - target,
                    "augment": 0,
                }
            else:
                needed = target - orig_count
                per_orig = min(max_per_orig - 1, max(1, needed // max(orig_count, 1)))
                remainder = needed - per_orig * orig_count
                plan[cls] = {
                    "action": "augment",
                    "orig": orig_count,
                    "target": target,
                    "per_orig": per_orig,
                    "remainder": max(0, remainder),
                    "augment": needed,
                }

        print(f"\n  {'Class':<20} {'Orig':>6} {'Target':>7} {'Action':>15} {'Detail'}")
        print(f"  {'-'*75}")
        for cls in CLASS_NAMES:
            p = plan[cls]
            if p["action"] == "undersample":
                print(f"  {cls:<20} {p['orig']:>6} {p['target']:>7} {'undersample':>15} "
                      f"remove {p['remove']} images")
            else:
                print(f"  {cls:<20} {p['orig']:>6} {p['target']:>7} {'augment':>15} "
                      f"+{p['augment']} (max {max_per_orig-1}x per image)")

        transform = build_strong_transform()
        mild_transform = build_mild_transform()

        log_rows = []

        for cls in CLASS_NAMES:
            p = plan[cls]
            label_files = class_labels_orig[cls]
            augmented_count = 0

            if p["action"] == "undersample":
                random.shuffle(label_files)
                files_to_remove = set(label_files[p["target"]:])
                for lf in files_to_remove:
                    img_path = resolve_image_path(train_img_dir, lf.stem)
                    if img_path and img_path.exists():
                        img_path.unlink()
                    if lf.exists():
                        lf.unlink()
                    total_orig -= 1
                print(f"  {cls}: removed {len(files_to_remove)} excess originals")
                log_rows.append({
                    "class": cls, "original": p["orig"],
                    "augmented": 0, "final": p["target"], "method": "undersample",
                })
            else:
                per_orig = p["per_orig"]
                remainder = p["remainder"]
                target_aug = p["augment"]
                skip_count = 0

                for lf in label_files:
                    image_path = resolve_image_path(train_img_dir, lf.stem)
                    if image_path is None:
                        skip_count += 1
                        continue
                    bboxes = load_yolo_bboxes(lf)
                    if not bboxes:
                        skip_count += 1
                        continue

                    num_copies = per_orig
                    if remainder > 0:
                        num_copies += 1
                        remainder -= 1

                    for copy_idx in range(min(num_copies, max_per_orig - 1)):
                        aug_img, aug_bboxes = apply_augmentation(image_path, bboxes, transform)
                        if aug_img is not None and aug_bboxes:
                            aug_uuid = f"aug_{cls}_{random.randint(100000, 999999)}"
                            aug_img_name = f"{aug_uuid}{image_path.suffix}"
                            aug_lbl_name = f"{aug_uuid}.txt"
                            cv2.imwrite(str(train_img_dir / aug_img_name), aug_img)
                            save_yolo_bboxes(train_lbl_dir / aug_lbl_name, aug_bboxes)
                            augmented_count += 1
                            if augmented_count >= target_aug:
                                break

                    if augmented_count >= target_aug:
                        break

                if skip_count > 0:
                    print(f"  {cls}: skipped {skip_count} files (couldn't read)")

                # If still short, try mosaics
                if augmented_count < target_aug:
                    extra_needed = target_aug - augmented_count
                    attempts = 0
                    max_attempts = extra_needed * 5
                    usable_files = [lf for lf in label_files if resolve_image_path(train_img_dir, lf.stem) is not None]

                    while extra_needed > 0 and attempts < max_attempts and len(usable_files) >= 2:
                        pair = random.sample(usable_files, 2)
                        pair_paths = []
                        pair_bboxes = []
                        skip = False
                        for p_file in pair:
                            p_img = resolve_image_path(train_img_dir, p_file.stem)
                            if p_img is None:
                                skip = True
                                break
                            pair_paths.append(p_img)
                            pair_bboxes.append(load_yolo_bboxes(p_file))
                        if skip or not all(pair_paths) or not all(pair_bboxes):
                            attempts += 1
                            continue

                        mosaic_img, mosaic_bboxes = build_mosaic(
                            pair_paths, pair_bboxes, mild_transform)
                        if mosaic_img is not None and mosaic_bboxes and len(mosaic_bboxes) > 0:
                            aug_uuid = f"aug_{cls}_{random.randint(100000, 999999)}"
                            aug_img_name = f"{aug_uuid}{pair_paths[0].suffix}"
                            aug_lbl_name = f"{aug_uuid}.txt"
                            cv2.imwrite(str(train_img_dir / aug_img_name), mosaic_img)
                            save_yolo_bboxes(train_lbl_dir / aug_lbl_name, mosaic_bboxes)
                            augmented_count += 1
                            extra_needed -= 1
                        attempts += 1

                total_augmented = augmented_count
                print(f"  {cls}: {p['orig']} originals → +{augmented_count} augmented = "
                      f"{p['orig'] + augmented_count} total (target: {target})")

                log_rows.append({
                    "class": cls,
                    "original": p["orig"],
                    "augmented": augmented_count,
                    "final": p["orig"] + augmented_count,
                    "method": "per-image + mosaic",
                })

        # Final verification
        print(f"\n{'='*60}")
        print(f"  POST-AUGMENTATION VERIFICATION")
        print(f"{'='*60}")
        final_counts = {}
        for cls in CLASS_NAMES:
            count = len(list(train_lbl_dir.glob(f"{cls}_*.txt")))
            count += len(list(train_lbl_dir.glob(f"aug_{cls}_*.txt")))
            final_counts[cls] = count
            print(f"  {cls:<20} {count:>6} images")

        total = sum(final_counts.values())
        min_c = min(final_counts.values())
        max_c = max(final_counts.values())
        print(f"\n  Total training images: {total}")
        print(f"  Range: {min_c} – {max_c}")
        print(f"  Imbalance ratio: {max_c/min_c:.2f}x (was 10.4x)")

        log_path = dataset_dir.parent / "results" / "augmentation_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "original", "augmented", "final", "method"])
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\n  Augmentation log saved: {log_path}")
        print("\n  Done! Dataset is now balanced.")


if __name__ == "__main__":
    main()