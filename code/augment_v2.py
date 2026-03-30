import argparse
import random
import uuid
from pathlib import Path

import albumentations as A
import cv2


RANDOM_SEED = 42
CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]
IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Apply offline augmentation to the v2 training split."
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(repo_root / "dataset"),
        help="Dataset directory containing train/valid/test folders.",
    )
    parser.add_argument(
        "--default-ratio",
        type=float,
        default=0.6,
        help="Fraction of each non-minority class to augment.",
    )
    parser.add_argument(
        "--minority-ratio",
        type=float,
        default=1.0,
        help="Fraction of minority-class images to augment.",
    )
    parser.add_argument(
        "--minority-classes",
        nargs="*",
        default=["fire_alarm", "wet_floor_sign"],
        help="Classes that should use the minority augmentation ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_yolo_bboxes(label_path: Path) -> list[list[float]]:
    bboxes: list[list[float]] = []
    if not label_path.exists():
        return bboxes

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            if len(coords) == 4:
                cx, cy, width, height = coords
                bboxes.append([cls, cx, cy, width, height])
            elif len(coords) >= 8:
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                bboxes.append([cls, cx, cy, width, height])

    return bboxes


def save_yolo_bboxes(label_path: Path, bboxes: list[list[float]]) -> None:
    with label_path.open("w", encoding="utf-8") as handle:
        for bbox in bboxes:
            handle.write(
                f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n"
            )


def clamp_bbox(
    cx: float, cy: float, width: float, height: float
) -> tuple[float, float, float, float]:
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    width = max(width, 0.01)
    height = max(height, 0.01)

    if cx - width / 2 < 0:
        width = cx * 2
    if cx + width / 2 > 1:
        width = (1 - cx) * 2
    if cy - height / 2 < 0:
        height = cy * 2
    if cy + height / 2 > 1:
        height = (1 - cy) * 2

    return cx, cy, width, height


def yolo_to_albumentations(
    bboxes: list[list[float]], img_w: int, img_h: int
) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    classes: list[int] = []

    for bbox in bboxes:
        cls, cx, cy, width, height = bbox
        x1 = (cx - width / 2) * img_w
        y1 = (cy - height / 2) * img_h
        x2 = (cx + width / 2) * img_w
        y2 = (cy + height / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        classes.append(int(cls))

    return boxes, classes


def albumentations_to_yolo(
    boxes: list[list[float]], classes: list[int], img_w: int, img_h: int
) -> list[list[float]]:
    bboxes: list[list[float]] = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        cx, cy, width, height = clamp_bbox(cx, cy, width, height)
        if width > 0.01 and height > 0.01:
            bboxes.append([cls, cx, cy, width, height])
    return bboxes


def resolve_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_transform() -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.ToGray(p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=1.0
            ),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.3, label_fields=["class_labels"]
        ),
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"
    minority_classes = set(args.minority_classes)

    if not train_img_dir.exists() or not train_lbl_dir.exists():
        raise FileNotFoundError(f"Training split not found under {dataset_dir}")

    transform = build_transform()

    print("Loading training data...")
    print(f"dataset: {dataset_dir}")

    class_labels: dict[str, list[Path]] = {class_name: [] for class_name in CLASS_NAMES}
    for label_file in sorted(train_lbl_dir.glob("*.txt")):
        if label_file.stem.startswith("aug_"):
            continue
        for class_name in CLASS_NAMES:
            if label_file.stem.startswith(class_name):
                class_labels[class_name].append(label_file)
                break

    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {len(class_labels[class_name])} images")

    print("\nApplying offline augmentation...")
    print(f"  default ratio: {args.default_ratio:.0%}")
    print(f"  minority ratio: {args.minority_ratio:.0%}")
    print(f"  minority classes: {sorted(minority_classes)}")

    total_augmented = 0
    for class_name in CLASS_NAMES:
        label_files = class_labels[class_name]
        if not label_files:
            continue

        ratio = (
            args.minority_ratio
            if class_name in minority_classes
            else args.default_ratio
        )
        target_count = int(len(label_files) * ratio)
        shuffled_indices = list(range(len(label_files)))
        random.shuffle(shuffled_indices)
        selected_indices = set(shuffled_indices[:target_count])

        originals = 0
        augmented = 0

        for idx, label_file in enumerate(label_files):
            if idx not in selected_indices:
                continue

            image_path = resolve_image_path(train_img_dir, label_file.stem)
            if image_path is None:
                continue

            bboxes = load_yolo_bboxes(label_file)
            if not bboxes:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            height, width = image.shape[:2]
            alb_boxes, alb_classes = yolo_to_albumentations(bboxes, width, height)

            try:
                result = transform(
                    image=image, bboxes=alb_boxes, class_labels=alb_classes
                )
                aug_bboxes = albumentations_to_yolo(
                    result["bboxes"], result["class_labels"], width, height
                )
            except Exception as exc:
                print(
                    f"    {class_name}: skipped one file due to augmentation error: {exc}"
                )
                continue

            if not aug_bboxes:
                continue

            aug_uuid = uuid.uuid4().hex[:8]
            aug_img_name = f"aug_{class_name}_{aug_uuid}{image_path.suffix}"
            aug_lbl_name = f"aug_{class_name}_{aug_uuid}.txt"

            cv2.imwrite(str(train_img_dir / aug_img_name), result["image"])
            save_yolo_bboxes(train_lbl_dir / aug_lbl_name, aug_bboxes)

            originals += 1
            augmented += 1

        total_augmented += augmented
        print(
            f"  {class_name}: {originals} -> {augmented} augmented (of {len(label_files)} total)"
        )

    print(f"\nTotal augmented: {total_augmented}")
    print("\n=== Final Stats ===")
    for class_name in CLASS_NAMES:
        original_count = len(list(train_lbl_dir.glob(f"{class_name}_*.txt")))
        augmented_count = len(list(train_lbl_dir.glob(f"aug_{class_name}_*.txt")))
        print(
            f"  {class_name}: {original_count} original + {augmented_count} augmented = {original_count + augmented_count} total"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
