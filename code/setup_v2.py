import argparse
import random
import shutil
import uuid
import zipfile
from pathlib import Path

import yaml


RANDOM_SEED = 42
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]

DATASETS = [
    {
        "zip_name": "wet-floor-detection1.v2i.yolov8.zip",
        "target_id": 0,
        "target_name": "wet_floor_sign",
        "keep_substrings": ["wet floor sign"],
    },
    {
        "zip_name": "Fire Alarm.v24i.yolov8 (1).zip",
        "target_id": 1,
        "target_name": "fire_alarm",
        "keep_substrings": ["fire alarm"],
    },
    {
        "zip_name": "Emergency Exit Signs.v4i.yolov8.zip",
        "target_id": 2,
        "target_name": "emergency_exit",
        "keep_substrings": ["exit"],
    },
    {
        "zip_name": "Hard Hat Universe.v4i.yolov8.zip",
        "target_id": 3,
        "target_name": "safety_helmet",
        "keep_substrings": ["helmet"],
    },
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build the v2 YOLO dataset from source zip files.")
    parser.add_argument(
        "--source-dir",
        default=str(repo_root),
        help="Directory containing the source zip files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "dataset"),
        help="Directory where the rebuilt dataset should be written.",
    )
    return parser.parse_args()


def load_yaml_from_zip(zip_file: zipfile.ZipFile) -> list[str]:
    yaml_name = next((name for name in zip_file.namelist() if name.endswith("data.yaml")), None)
    if yaml_name is None:
        raise FileNotFoundError("No data.yaml found in zip archive.")

    data = yaml.safe_load(zip_file.read(yaml_name)) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[idx] for idx in sorted(names)]
    return [str(name) for name in names]


def resolve_keep_ids(class_names: list[str], keep_substrings: list[str]) -> set[int]:
    keep_ids = set()
    for idx, class_name in enumerate(class_names):
        lowered = class_name.lower()
        if any(fragment in lowered for fragment in keep_substrings):
            keep_ids.add(idx)
    return keep_ids


def collect_examples(zip_path: Path, keep_ids: set[int]) -> list[tuple[bytes, str, list[str]]]:
    examples: list[tuple[bytes, str, list[str]]] = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for image_name in zip_file.namelist():
            suffix = Path(image_name).suffix.lower()
            if suffix not in IMG_EXTENSIONS or "/images/" not in image_name:
                continue

            label_name = image_name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            try:
                label_text = zip_file.read(label_name).decode("utf-8")
            except KeyError:
                continue

            kept_boxes: list[str] = []
            for raw_line in label_text.splitlines():
                parts = raw_line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                if class_id in keep_ids:
                    kept_boxes.append(" ".join(parts[1:]))

            if kept_boxes:
                examples.append((zip_file.read(image_name), suffix, kept_boxes))

    return examples


def ensure_clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "valid", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def write_split(
    output_dir: Path,
    split_name: str,
    target_id: int,
    target_name: str,
    examples: list[tuple[bytes, str, list[str]]],
) -> int:
    written = 0
    for image_bytes, suffix, boxes in examples:
        stem = f"{target_name}_{uuid.uuid4().hex[:8]}"
        image_path = output_dir / split_name / "images" / f"{stem}{suffix}"
        label_path = output_dir / split_name / "labels" / f"{stem}.txt"

        image_path.write_bytes(image_bytes)
        with label_path.open("w", encoding="utf-8") as handle:
            for box in boxes:
                handle.write(f"{target_id} {box}\n")
        written += 1
    return written


def write_data_yaml(output_dir: Path) -> None:
    data_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    random.seed(RANDOM_SEED)
    ensure_clean_output(output_dir)

    per_class_examples: dict[int, list[tuple[bytes, str, list[str]]]] = {idx: [] for idx in range(len(CLASS_NAMES))}

    for dataset in DATASETS:
        zip_path = source_dir / dataset["zip_name"]
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing zip file: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            source_class_names = load_yaml_from_zip(zip_file)

        keep_ids = resolve_keep_ids(source_class_names, dataset["keep_substrings"])
        if not keep_ids:
            raise RuntimeError(
                f"No matching classes found in {dataset['zip_name']}. "
                f"Expected one of {dataset['keep_substrings']}, found {source_class_names}."
            )

        examples = collect_examples(zip_path, keep_ids)
        per_class_examples[dataset["target_id"]].extend(examples)

        print(f"\n=== {dataset['target_name']} ===")
        print(f"zip: {dataset['zip_name']}")
        print(f"source classes: {source_class_names}")
        print(f"kept source IDs: {sorted(keep_ids)}")
        print(f"examples collected: {len(examples)}")

    print("\n=== Split Summary ===")
    total_written = 0
    for target_id, target_name in enumerate(CLASS_NAMES):
        examples = per_class_examples[target_id]
        random.shuffle(examples)

        total = len(examples)
        train_cutoff = int(total * 0.7)
        valid_cutoff = train_cutoff + int(total * 0.2)

        train_examples = examples[:train_cutoff]
        valid_examples = examples[train_cutoff:valid_cutoff]
        test_examples = examples[valid_cutoff:]

        train_count = write_split(output_dir, "train", target_id, target_name, train_examples)
        valid_count = write_split(output_dir, "valid", target_id, target_name, valid_examples)
        test_count = write_split(output_dir, "test", target_id, target_name, test_examples)
        total_written += train_count + valid_count + test_count

        print(
            f"{target_name}: total={total} train={train_count} valid={valid_count} test={test_count}"
        )

    write_data_yaml(output_dir)

    print("\n=== Done ===")
    print(f"output: {output_dir}")
    print(f"total examples written: {total_written}")


if __name__ == "__main__":
    main()
