#!/usr/bin/env python3
"""
Compute inverse-frequency class weights from training labels.
Output: dataset/class_weights.yaml
Useful for custom training or loss function weighting.
"""

import yaml
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]


def compute_weights(dataset_dir="dataset"):
    dataset_path = Path(dataset_dir)
    labels_dir = dataset_path / "train" / "labels"

    if not labels_dir.exists():
        print(f"Error: {labels_dir} not found.")
        return

    class_counts = defaultdict(int)
    total_boxes = 0

    for label_file in labels_dir.glob("*.txt"):
        try:
            content = label_file.read_text().strip()
            for line in content.split('\n'):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if 0 <= cls_id < len(CLASS_NAMES):
                    class_counts[cls_id] += 1
                    total_boxes += 1
        except Exception:
            continue

    print(f"\n{'='*50}")
    print(f"  CLASS WEIGHT COMPUTATION")
    print(f"{'='*50}")
    print(f"  Total bounding boxes: {total_boxes}")
    print(f"\n  {'Class':<20} {'Count':>8} {'Weight':>10}")
    print(f"  {'-'*42}")

    # Compute inverse-frequency weights, normalize so max weight = 1.0
    weights = {}
    max_count = max(class_counts.values()) if class_counts else 1

    for cls_id in range(len(CLASS_NAMES)):
        count = class_counts.get(cls_id, 0)
        if count > 0:
            # Inverse frequency, normalized
            weight = round(max_count / count, 4)
        else:
            weight = 1.0
        weights[cls_id] = weight
        print(f"  {CLASS_NAMES[cls_id]:<20} {count:>8} {weight:>10.4f}")

    # Save YAML
    output = dataset_path / "class_weights.yaml"
    data = {
        "class_weights": weights,
        "note": (
            "Computed from training split labels. "
            "Multiply each class's loss contribution by its weight during training. "
            "Use with caution — equalized datasets render these unnecessary."
        ),
    }
    with open(output, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Weights saved to: {output}")
    return weights


if __name__ == "__main__":
    compute_weights()