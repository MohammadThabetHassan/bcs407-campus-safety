#!/usr/bin/env python3
"""
Deep quantitative analysis of dataset: bbox sizes, aspect ratios.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]
CLASS_LABELS = [c.replace("_", " ").title() for c in CLASS_NAMES]


def analyze_bbox_stats(dataset_dir="dataset"):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: {dataset_dir} not found.")
        sys.exit(1)

    stats = {}

    for split in ("train", "valid", "test"):
        labels_dir = dataset_path / split / "labels"
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} not found, skipping.")
            continue

        classes_data = {
            c: {"areas": [], "aspect_ratios": [], "widths": [], "heights": []}
            for c in CLASS_NAMES
        }
        file_count = 0

        for label_file in sorted(labels_dir.glob("*.txt")):
            file_count += 1
            try:
                content = label_file.read_text().strip()
                if not content:
                    continue
                for line in content.split('\n'):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id >= len(CLASS_NAMES):
                        continue
                    w, h = float(parts[3]), float(parts[4])
                    cls_name = CLASS_NAMES[cls_id]
                    area = w * h
                    ar = w / h if h > 0 else 0

                    classes_data[cls_name]["widths"].append(w)
                    classes_data[cls_name]["heights"].append(h)
                    classes_data[cls_name]["areas"].append(area)
                    classes_data[cls_name]["aspect_ratios"].append(ar)
            except Exception:
                pass

        stats[split] = {
            "total_files": file_count,
            "total_bboxes": sum(len(classes_data[c]["areas"]) for c in CLASS_NAMES),
            "per_class": {},
        }

        print(f"\n--- {split.upper()} SPLIT ---")
        print(f"Label files: {file_count}")
        print(f"Total bboxes: {stats[split]['total_bboxes']}")

        for cls in CLASS_NAMES:
            d = classes_data[cls]
            count = len(d["areas"])
            if count == 0:
                stats[split]["per_class"][cls] = {"count": 0}
                print(f"  {cls:<20} {0:>8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>8}")
                continue

            avg_w = sum(d["widths"]) / count
            avg_h = sum(d["heights"]) / count
            avg_area = sum(d["areas"]) / count
            avg_ar = sum(d["aspect_ratios"]) / count

            stats[split]["per_class"][cls] = {
                "count": count,
                "avg_width": round(avg_w, 4),
                "avg_height": round(avg_h, 4),
                "avg_area": round(avg_area, 4),
                "avg_aspect_ratio": round(avg_ar, 4),
                "min_area": round(min(d["areas"]), 4),
                "max_area": round(max(d["areas"]), 4),
            }

            print(f"{cls:<20} {count:>8} {avg_w:>8.4f} {avg_h:>8.4f} "
                  f"{avg_area:>10.4f} {avg_ar:>8.3f}")

    with open(dataset_path / "bbox_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nBbox stats saved to: {dataset_path / 'bbox_stats.json'}")

    # Generate charts
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_dir = Path("results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        colors = {
            "wet_floor_sign": "#FF6B6B",
            "fire_alarm": "#FFA726",
            "emergency_exit": "#66BB6A",
            "safety_helmet": "#42A5F5",
        }

        # Raw area + AR data for charting
        tl_dir = dataset_path / "train" / "labels"
        raw_areas = {c: [] for c in CLASS_NAMES}
        raw_ars = {c: [] for c in CLASS_NAMES}
        for label_file in tl_dir.glob("*.txt"):
            try:
                for line in label_file.read_text().strip().split('\n'):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if 0 <= cls_id < len(CLASS_NAMES):
                        w, h = float(parts[3]), float(parts[4])
                        raw_areas[CLASS_NAMES[cls_id]].append(w * h)
                        raw_ars[CLASS_NAMES[cls_id]].append(w / h if h > 0 else 0)
            except Exception:
                pass

        # --- Chart: Bbox area distribution ---
        train_data = stats.get("train", {}).get("per_class", {})
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, cls in enumerate(CLASS_NAMES):
            areas = raw_areas[cls]
            if areas:
                axes[i].hist(areas, bins=40, color=colors[cls],
                             edgecolor='white', linewidth=0.5, alpha=0.85)
                avg = sum(areas) / len(areas)
                axes[i].axvline(avg, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {avg:.4f}')
                axes[i].legend(fontsize=9)
            axes[i].set_title(cls.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            axes[i].set_xlabel('Bounding Box Area (relative)', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        fig.suptitle('Bounding Box Area Distribution — Training Split', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / 'bbox_area_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Chart: {output_dir / 'bbox_area_distribution.png'}")

        # --- Chart: Aspect ratio distribution ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, cls in enumerate(CLASS_NAMES):
            ars = raw_ars[cls]
            if ars:
                axes[i].hist(ars, bins=30, color=colors[cls],
                             edgecolor='white', linewidth=0.5, alpha=0.85)
                avg_ar = sum(ars) / len(ars)
                axes[i].axvline(avg_ar, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {avg_ar:.2f}')
                axes[i].legend(fontsize=9)
            axes[i].set_title(cls.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            axes[i].set_xlabel('Aspect Ratio (w/h)', fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        fig.suptitle('Aspect Ratio Distribution — Training Split', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / 'aspect_ratio_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Chart: {output_dir / 'aspect_ratio_distribution.png'}")

        # --- Chart: Average bbox size comparison ---
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_areas = [train_data.get(c, {}).get("avg_area", 0) for c in CLASS_NAMES]
        bars = ax.bar(range(len(CLASS_NAMES)), avg_areas, width=0.6,
                     color=[colors[c] for c in CLASS_NAMES], edgecolor='white')
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_LABELS, fontsize=11)
        ax.set_ylabel('Average Bounding Box Area', fontsize=12)
        ax.set_title('Comparison of Average BBox Sizes by Class', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, avg_areas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                   f'{val:.5f}', ha='center', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / 'avg_bbox_size_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Chart: {output_dir / 'avg_bbox_size_comparison.png'}")

        # --- Chart: Bbox count per split ---
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(CLASS_NAMES))
        width = 0.25
        for i, split in enumerate(("train", "valid", "test")):
            counts = [stats.get(split, {}).get("per_class", {}).get(c, {}).get("count", 0)
                     for c in CLASS_NAMES]
            ax.bar([p + width*i for p in x], counts, width,
                   label=split.capitalize(), color=[colors[c] for c in CLASS_NAMES],
                   alpha=0.75, edgecolor='white')
        ax.set_xticks([p + width for p in x])
        ax.set_xticklabels(CLASS_LABELS, fontsize=10)
        ax.set_ylabel('Bbox Count', fontsize=12)
        ax.set_title('Bounding Box Count by Class and Split', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / 'bbox_count_splits.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Chart: {output_dir / 'bbox_count_splits.png'}")

        print("\n  All charts generated!")

    except ImportError:
        print("\n  [INFO] matplotlib not available — skipping charts")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    if not Path(ds).exists():
        print("Error: dataset not found. Run analyze_distribution.py --demo first.")
        sys.exit(1)
    print(f"\nAnalyzing bbox statistics: {ds}")
    analyze_bbox_stats(ds)