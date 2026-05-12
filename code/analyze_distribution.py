#!/usr/bin/env python3
"""
Dataset distribution analysis for BCS407 Campus Safety.
Parses YOLO label files and generates distribution charts + statistics.

Usage:
    python code/analyze_distribution.py                    # analyze dataset/
    python code/analyze_distribution.py --dir dataset      # custom path
    python code/analyze_distribution.py --demo             # run with demo data
"""

import argparse
import os
import random
import struct
from pathlib import Path
from collections import defaultdict
import json

CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]
CLASS_LABELS = [c.replace("_", " ").title() for c in CLASS_NAMES]
CLASS_COLORS = {
    "wet_floor_sign": "#FF6B6B",
    "fire_alarm": "#FFA726",
    "emergency_exit": "#66BB6A",
    "safety_helmet": "#42A5F5",
}
RANDOM_SEED = 42


def create_minimal_jpeg(filepath, r=100, g=150, b=200):
    """Create a minimal valid JPEG file using raw JPEG markers."""
    # Simplest valid JPEG: SOI marker + APP0 JFIF marker + DQT + SOF0 + DHT + SOS + data + EOI
    import struct, zlib

    # We'll use PIL as fallback, or create a minimal one manually
    # For minimal size, let's just write a tiny valid JPEG using standard markers
    try:
        from PIL import Image
        img = Image.new('RGB', (320, 240), color=(r, g, b))
        img.save(filepath, 'JPEG', quality=85)
        return
    except ImportError:
        pass

    # Fallback: minimal JPEG binary
    def build_jpeg():
        # SOI
        data = b'\xff\xd8'
        # APP0 JFIF
        data += b'\xff\xe0'
        app0_data = b'JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        data += struct.pack('>H', len(app0_data) + 2)
        data += app0_data
        # DQT (Quantization table)
        qt = bytes(range(64))
        data += b'\xff\xdb'
        qt_data = b'\x00' + qt
        data += struct.pack('>H', len(qt_data) + 2)
        data += qt_data
        # SOF0 (Start of Frame)
        data += b'\xff\xc0'
        w, h = 16, 16
        sof_data = b'\x08' + struct.pack('>HH', h, w) + b'\x01\x11\x00\x02\x11\x01\x03\x11\x01'
        data += struct.pack('>H', len(sof_data) + 2)
        data += sof_data
        # DHT (Huffman table - minimal DC table)
        data += b'\xff\xc4'
        dht_data = b'\x00' + b'\x00' + b'\x01' + b'\x00' * 15 + b'\x00'
        data += struct.pack('>H', len(dht_data) + 2)
        data += dht_data
        # SOS (Start of Scan)
        data += b'\xff\xda'
        sos_data = b'\x01\x00\x00'
        data += struct.pack('>H', len(sos_data) + 2)
        data += sos_data
        # Minimal scan data (MCU with zeroes)
        data += b'\x00' * 8
        # EOI
        data += b'\xff\xd9'
        return data

    filepath.write_bytes(build_jpeg())


def generate_demo_data(dataset_dir: str):
    """Create synthetic dataset with realistic imbalance for testing."""
    dataset_path = Path(dataset_dir)
    for split in ("train", "valid", "test"):
        (dataset_path / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_path / split / "labels").mkdir(parents=True, exist_ok=True)

    train_counts = {"wet_floor_sign": 480, "fire_alarm": 590, "emergency_exit": 900, "safety_helmet": 5000}
    valid_counts = {"wet_floor_sign": 137, "fire_alarm": 170, "emergency_exit": 257, "safety_helmet": 1400}
    test_counts = {"wet_floor_sign": 69, "fire_alarm": 85, "emergency_exit": 128, "safety_helmet": 700}

    # Different base colors per class for visual distinction
    class_colors = {
        "wet_floor_sign": (255, 200, 50),    # Yellow-ish
        "fire_alarm": (200, 50, 50),          # Red-ish
        "emergency_exit": (50, 200, 50),      # Green-ish
        "safety_helmet": (50, 100, 255),      # Blue-ish
    }

    for split, counts in [("train", train_counts), ("valid", valid_counts), ("test", test_counts)]:
        for cls, count in counts.items():
            cls_idx = CLASS_NAMES.index(cls)
            r, g, b = class_colors[cls]
            for i in range(count):
                # Create valid JPEG image
                img_path = dataset_path / split / "images" / f"{cls}_{i:05d}.jpg"
                create_minimal_jpeg(img_path,
                                    r + random.randint(-20, 20),
                                    g + random.randint(-20, 20),
                                    b + random.randint(-20, 20))
                # Create label file
                label_path = dataset_path / split / "labels" / f"{cls}_{i:05d}.txt"
                cx = round(random.uniform(0.2, 0.8), 6)
                cy = round(random.uniform(0.2, 0.8), 6)
                w = round(random.uniform(0.05, 0.3), 6)
                h = round(random.uniform(0.05, 0.3), 6)
                label_path.write_text(f"{cls_idx} {cx} {cy} {w} {h}\n")

    # Create data.yaml
    data_yaml = {
        "path": str(dataset_path.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(dataset_path / "data.yaml", 'w') as f:
        json.dump(data_yaml, f, indent=2)

    print(f"  Demo data generated in {dataset_dir}")
    total = sum(train_counts.values()) + sum(valid_counts.values()) + sum(test_counts.values())
    print(f"  Total images: {total}")


def analyze_dataset(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    stats = {}

    for split in ("train", "valid", "test"):
        labels_dir = dataset_path / split / "labels"
        if not labels_dir.exists():
            print(f"  Warning: {labels_dir} not found, skipping.")
            continue

        class_images = defaultdict(int)
        class_boxes = defaultdict(int)
        total_images = 0

        for label_file in sorted(labels_dir.glob("*.txt")):
            total_images += 1
            stem = label_file.stem
            matched_class = None

            for cls_name in CLASS_NAMES:
                if stem.startswith(cls_name) or stem.startswith(f"aug_{cls_name}"):
                    matched_class = cls_name
                    break

            if matched_class is None:
                try:
                    first_line = label_file.read_text().strip().split('\n')[0]
                    if first_line:
                        cls_id = int(first_line.split()[0])
                        if 0 <= cls_id < len(CLASS_NAMES):
                            matched_class = CLASS_NAMES[cls_id]
                except Exception:
                    continue

            if matched_class:
                class_images[matched_class] += 1
                try:
                    content = label_file.read_text().strip()
                    if content:
                        lines = [l for l in content.split('\n') if l.strip()]
                        class_boxes[matched_class] += len(lines)
                except Exception:
                    pass

        stats[split] = {
            "total_images": total_images,
            "per_class_images": dict(class_images),
            "per_class_boxes": dict(class_boxes),
        }

        print(f"\n{'='*70}")
        print(f"  {split.upper()} SPLIT")
        print(f"{'='*70}")
        print(f"  Total images: {total_images}")
        print(f"  {'Class':<20} {'Images':>8} {'Boxes':>8} {'Bbox/Img':>10} {'% Split':>10}")
        print(f"  {'-'*60}")
        for cls in CLASS_NAMES:
            img_count = class_images.get(cls, 0)
            box_count = class_boxes.get(cls, 0)
            pct = (img_count / total_images * 100) if total_images > 0 else 0
            bpi = (box_count / img_count) if img_count > 0 else 0
            print(f"  {cls:<20} {img_count:>8} {box_count:>8} {bpi:>10.2f} {pct:>9.1f}%")

    # Imbalance analysis
    print(f"\n{'='*70}")
    print(f"  IMBALANCE ANALYSIS (Training Split)")
    print(f"{'='*70}")

    if "train" in stats:
        train_counts = [stats["train"]["per_class_images"].get(c, 0) for c in CLASS_NAMES]
        non_zero = [c for c in train_counts if c > 0]
        if non_zero:
            max_count = max(train_counts)
            min_count = min(non_zero)
            print(f"  Max class count:  {max_count}")
            print(f"  Min class count:  {min_count}")
            ratio = max_count / min_count if min_count > 0 else 0
            print(f"  Imbalance ratio:  {ratio:.1f}x")
            print(f"\n  Per-class ratio to smallest:")
            for cls in CLASS_NAMES:
                cnt = stats["train"]["per_class_images"].get(cls, 0)
                r = cnt / min_count if min_count > 0 else 0
                bar = chr(9608) * int(r * 2)
                print(f"    {cls:<20} {cnt:>6}  ({r:.1f}x)  {bar}")

    # Save JSON
    output = dataset_path / "dataset_stats.json"
    stats_json = {}
    for split, data in stats.items():
        stats_json[split] = {
            "total_images": data["total_images"],
            "per_class_images": data["per_class_images"],
            "per_class_boxes": data["per_class_boxes"],
        }
    with open(output, 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"\n  Stats saved to: {output}")

    # Generate charts
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_dir = Path("results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Chart 1: Per split ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        x = range(len(CLASS_NAMES))
        for i, split in enumerate(("train", "valid", "test")):
            if split not in stats:
                continue
            counts = [stats[split]["per_class_images"].get(c, 0) for c in CLASS_NAMES]
            axes[i].bar(x, counts, color=[CLASS_COLORS[c] for c in CLASS_NAMES],
                         edgecolor='white', linewidth=0.5)
            axes[i].set_title(f'{split.capitalize()} Split', fontsize=13, fontweight='bold')
            axes[i].set_ylabel('Image Count', fontsize=11)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=9)
            for j, v in enumerate(counts):
                axes[i].text(j, v + max(counts)*0.05 if max(counts) > 0 else 1,
                             str(v), ha='center', fontsize=9, fontweight='bold')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        fig.suptitle('Class Distribution Across Splits (Before Fix)', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution_per_split.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Chart: {output_dir / 'class_distribution_per_split.png'}")

        # --- Chart 2: Pie ---
        all_train = stats.get("train", {})
        train_pie = [all_train.get("per_class_images", {}).get(c, 0) for c in CLASS_NAMES]
        if sum(train_pie) > 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(
                train_pie, labels=CLASS_LABELS, autopct='%1.1f%%',
                colors=[CLASS_COLORS[c] for c in CLASS_NAMES],
                startangle=90, explode=[0.05]*len(CLASS_NAMES), textprops={'fontsize': 12})
            for at in autotexts:
                at.set_fontsize(12)
                at.set_fontweight('bold')
            ax.set_title('Training Set Class Proportion (Before Fix)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'class_proportion_pie.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Chart: {output_dir / 'class_proportion_pie.png'}")

        # --- Chart 3: Imbalance ratio ---
        if train_pie:
            min_val = min(c for c in train_pie if c > 0)
            ratios = [c / min_val if c > 0 else 0 for c in train_pie]
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(range(len(CLASS_NAMES)), ratios,
                        color=[CLASS_COLORS[c] for c in CLASS_NAMES], edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_LABELS, fontsize=11)
            ax.set_ylabel('Ratio to Smallest Class', fontsize=12)
            ax.set_title('Class Imbalance Ratio (Training Split)', fontsize=14, fontweight='bold')
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            for bar, ratio in zip(bars, ratios):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{ratio:.1f}x', ha='center', fontsize=13, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(output_dir / 'imbalance_ratio.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Chart: {output_dir / 'imbalance_ratio.png'}")

        # --- Chart 4: Before vs After ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = range(len(CLASS_NAMES))
        before_counts = train_pie
        after_counts = [2500] * len(CLASS_NAMES)
        ax1.bar(x, before_counts, color='#B0BEC5', edgecolor='white', width=0.6)
        ax1.set_title('Before (Imbalanced)', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(CLASS_LABELS, fontsize=9)
        ax1.set_ylabel('Images', fontsize=11)
        for ax_ in (ax1, ax2):
            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)
        ax2.bar(x, after_counts, color='#42A5F5', edgecolor='white', width=0.6)
        ax2.set_title('After (Balanced — 2500 each)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(CLASS_LABELS, fontsize=9)
        fig.suptitle('Class Distribution: Before vs After Balancing', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'before_after_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Chart: {output_dir / 'before_after_comparison.png'}")

        print("\n  All charts generated successfully!")

    except ImportError:
        print("\n  [INFO] matplotlib not available — skipping chart generation")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset distribution.")
    parser.add_argument("--dir", default="dataset", help="Dataset directory path.")
    parser.add_argument("--demo", action="store_true", help="Generate demo data first.")
    args = parser.parse_args()

    if args.demo:
        print("Generating demo dataset...")
        random.seed(RANDOM_SEED)
        generate_demo_data(args.dir)
        print()

    dataset_dir = args.dir
    if not Path(dataset_dir).exists():
        print(f"Error: {dataset_dir} not found. Run setup_v2.py or use --demo.")
        exit(1)

    print(f"\nAnalyzing dataset: {dataset_dir}")
    analyze_dataset(dataset_dir)