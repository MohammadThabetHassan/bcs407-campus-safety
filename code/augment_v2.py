import os
import yaml
import random
import uuid
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "campus_safety_v2"
TRAIN_IMG = DATA_DIR / "train" / "images"
TRAIN_LBL = DATA_DIR / "train" / "labels"

CLASS_NAMES = ['wet_floor_sign', 'fire_alarm', 'emergency_exit', 'safety_helmet']

def load_yolo_bboxes(label_path):
    bboxes = []
    if not label_path.exists():
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            if len(coords) == 4:
                cx, cy, w, h = coords
                bboxes.append([cls, cx, cy, w, h])
            elif len(coords) >= 8:
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min
                bboxes.append([cls, cx, cy, w, h])
    
    return bboxes

def save_yolo_bboxes(label_path, bboxes):
    with open(label_path, 'w') as f:
        for bb in bboxes:
            f.write(f"{int(bb[0])} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f} {bb[4]:.6f}\n")

def clamp_bbox(cx, cy, w, h):
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    if w < 0.01:
        w = 0.01
    if h < 0.01:
        h = 0.01
    if cx - w/2 < 0:
        w = cx * 2
    if cx + w/2 > 1:
        w = (1 - cx) * 2
    if cy - h/2 < 0:
        h = cy * 2
    if cy + h/2 > 1:
        h = (1 - cy) * 2
    return cx, cy, w, h

def yolo_toAlbumentation(bboxes, img_w, img_h):
    boxes = []
    classes = []
    for bb in bboxes:
        cls = bb[0]
        cx, cy, w, h = bb[1], bb[2], bb[3], bb[4]
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
    return boxes, classes

def albumentation_toYolo(boxes, classes, img_w, img_h):
    bboxes = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cx, cy, w, h = clamp_bbox(cx, cy, w, h)
        if w > 0.01 and h > 0.01:
            bboxes.append([cls, cx, cy, w, h])
    return bboxes

augment_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
    A.ToGray(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=1.0),
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
    A.HorizontalFlip(p=0.5),
], A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

print("Loading training data...")

class_images = {cn: [] for cn in CLASS_NAMES}
for lbl_file in TRAIN_LBL.glob("*.txt"):
    if lbl_file.stem.startswith('aug_'):
        continue
    stem = lbl_file.stem
    for cn in CLASS_NAMES:
        if stem.startswith(cn):
            class_images[cn].append(lbl_file)
            break

for cn in CLASS_NAMES:
    class_images[cn] = sorted(class_images[cn])
    print(f"  {cn}: {len(class_images[cn])} images")

print("\nApplying offline augmentation...")
print("  - 60% of training images (random selection)")
print("  - 100% of fire_alarm and wet_floor_sign (minority classes)")

all_augmented = 0
for cn in CLASS_NAMES:
    lbl_files = class_images[cn]
    if not lbl_files:
        continue
    
    should_augment_pct = 1.0 if cn in ['fire_alarm', 'wet_floor_sign'] else 0.6
    n_to_augment = int(len(lbl_files) * should_augment_pct)
    
    all_idxs = list(range(len(lbl_files)))
    random.shuffle(all_idxs)
    idxs_to_augment = set(all_idxs[:n_to_augment])
    
    originals = 0
    augmented = 0
    
    for idx, lbl_file in enumerate(lbl_files):
        img_stem = lbl_file.stem
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            candidate = TRAIN_IMG / (img_stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        
        if not img_file:
            continue
        
        bboxes = load_yolo_bboxes(lbl_file)
        if not bboxes:
            continue
        
        if idx in idxs_to_augment:
            originals += 1
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            a_boxes, a_classes = yolo_toAlbumentation(bboxes, w, h)
            
            try:
                augmented_result = augment_transform(image=img, bboxes=a_boxes, class_labels=a_classes)
                aug_bboxes = albumentation_toYolo(augmented_result['bboxes'], augmented_result['class_labels'], w, h)
                
                if aug_bboxes:
                    aug_uuid = uuid.uuid4().hex[:8]
                    aug_img_name = f"aug_{cn}_{aug_uuid}{img_file.suffix}"
                    aug_lbl_name = f"aug_{cn}_{aug_uuid}.txt"
                    
                    cv2.imwrite(str(TRAIN_IMG / aug_img_name), augmented_result['image'])
                    save_yolo_bboxes(TRAIN_LBL / aug_lbl_name, aug_bboxes)
                    augmented += 1
            except Exception as e:
                print(f"    Error: {e}")
    
    all_augmented += augmented
    print(f"  {cn}: {originals} -> {augmented} augmented (of {len(lbl_files)} total)")

print(f"\nTotal augmented: {all_augmented}")

print("\n=== FINAL STATS ===")
for cn in CLASS_NAMES:
    orig = len(list(TRAIN_LBL.glob(f"{cn}_*.txt")))
    aug = len(list(TRAIN_LBL.glob(f"aug_{cn}_*.txt")))
    print(f"  {cn}: {orig} original + {aug} augmented = {orig + aug} total")

print("\nDone!")