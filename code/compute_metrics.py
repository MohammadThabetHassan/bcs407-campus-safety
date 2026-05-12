#!/usr/bin/env python3
"""
Comprehensive metrics computation from YOLO validation results.
Reads results.csv from a YOLO run directory and produces per-class metrics
including F1-score, mAP gap analysis, and inference speed stats.

Usage:
    # From a YOLO run directory
    python code/compute_metrics.py --run-dir runs/detect/campus_safety_v3_balanced

    # From default best run
    python code/compute_metrics.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute detailed metrics from YOLO results.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to YOLO run directory (e.g., runs/detect/campus_safety_v3_balanced). "
             "If not given, searches for the most recent run.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Direct path to results.csv file.",
    )
    return parser.parse_args()


def find_results_csv(run_dir):
    """Find results.csv in the run directory."""
    run_path = Path(run_dir)
    # Check direct path
    csv_path = run_path / "results.csv"
    if csv_path.exists():
        return csv_path
    # Check nested paths
    for csv_file in run_path.rglob("results.csv"):
        return csv_file
    return None


def parse_results_csv(csv_path):
    """Parse YOLO results.csv into structured data."""
    epochs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(row)
    return epochs


def compute_best_epoch(epochs):
    """Find best epoch by mAP@0.5."""
    best = None
    best_map50 = 0
    for e in epochs:
        try:
            map50 = float(e.get("metrics/mAP50(B)", 0))
            if map50 > best_map50:
                best_map50 = map50
                best = e
        except (ValueError, TypeError):
            continue
    return best


def compute_final_epoch(epochs):
    """Get last epoch data."""
    return epochs[-1] if epochs else None


def extract_metrics(epoch_row):
    """Extract all metrics from an epoch row."""
    metrics = {}
    col_map = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
        "train_box_loss": "train/box_loss",
        "train_cls_loss": "train/cls_loss",
        "train_dfl_loss": "train/dfl_loss",
        "val_box_loss": "val/box_loss",
        "val_cls_loss": "val/cls_loss",
        "val_dfl_loss": "val/dfl_loss",
        "lr_pg0": "lr/pg0",
        "lr_pg1": "lr/pg1",
        "lr_pg2": "lr/pg2",
        "time": "time",
    }
    for key, col in col_map.items():
        try:
            metrics[key] = float(epoch_row.get(col, 0))
        except (ValueError, TypeError):
            metrics[key] = 0.0
    return metrics


def f1_score(precision, recall):
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def generate_overall_table(best, final):
    """Generate overall metrics comparison table."""
    print(f"\n{'='*80}")
    print(f"  OVERALL METRICS COMPARISON")
    print(f"{'='*80}")

    if best:
        b = extract_metrics(best)
        print(f"\n  Best Epoch (#{best.get('epoch', 'N/A')})")
        print(f"  {'Metric':<20} {'Value':>12}")
        print(f"  {'-'*35}")
        print(f"  {'Precision':<20} {b['precision']:>10.4f}")
        print(f"  {'Recall':<20} {b['recall']:>10.4f}")
        print(f"  {'F1-Score':<20} {f1_score(b['precision'], b['recall']):>10.4f}")
        print(f"  {'mAP@0.5':<20} {b['map50']:>10.4f}")
        print(f"  {'mAP@0.5:0.95':<20} {b['map50_95']:>10.4f}")
        print(f"  {'Train time (s)':<20} {b['time']:>10.1f}")

    if final:
        f = extract_metrics(final)
        print(f"\n  Final Epoch (#{final.get('epoch', 'N/A')})")
        print(f"  {'Metric':<20} {'Value':>12}")
        print(f"  {'-'*35}")
        print(f"  {'Precision':<20} {f['precision']:>10.4f}")
        print(f"  {'Recall':<20} {f['recall']:>10.4f}")
        print(f"  {'F1-Score':<20} {f1_score(f['precision'], f['recall']):>10.4f}")
        print(f"  {'mAP@0.5':<20} {f['map50']:>10.4f}")
        print(f"  {'mAP@0.5:0.95':<20} {f['map50_95']:>10.4f}")
        print(f"  {'Train time (s)':<20} {f['time']:>10.1f}")


def generate_training_curves(epochs, output_dir):
    """Generate training curve data and save as CSV for plotting."""
    curve_data = []
    for e in epochs:
        try:
            epoch_num = int(e.get("epoch", 0))
            metrics = extract_metrics(e)
            curve_data.append({
                "epoch": epoch_num,
                "train_box_loss": metrics["train_box_loss"],
                "train_cls_loss": metrics["train_cls_loss"],
                "train_dfl_loss": metrics["train_dfl_loss"],
                "val_box_loss": metrics["val_box_loss"],
                "val_cls_loss": metrics["val_cls_loss"],
                "val_dfl_loss": metrics["val_dfl_loss"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "lr": metrics["lr_pg0"],
                "time": metrics["time"],
            })
        except (ValueError, KeyError):
            continue

    # Save as processable CSV
    curve_path = output_dir / "training_curves_data.csv"
    if curve_data:
        with open(curve_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=curve_data[0].keys())
            writer.writeheader()
            writer.writerows(curve_data)
        print(f"  Curves data: {curve_path}")

    return curve_data


def print_convergence_analysis(curve_data):
    """Analyze when the model converged."""
    if len(curve_data) < 5:
        print("\n  Insufficient data for convergence analysis.")
        return

    # Find convergence: val_loss plateaus (change < 1% over 10 epochs)
    val_losses = [d["val_box_loss"] + d["val_cls_loss"] + d["val_dfl_loss"] for d in curve_data]
    epochs_list = [d["epoch"] for d in curve_data]

    convergence_epoch = None
    for i in range(10, len(val_losses)):
        window = val_losses[i-10:i]
        avg = sum(window) / len(window)
        if avg > 0:
            max_change = max(abs(v - avg) / avg for v in window)
            if max_change < 0.01:  # Less than 1% variation
                convergence_epoch = epochs_list[i-10]
                break

    map50_values = [d["map50"] for d in curve_data]
    best_map50 = max(map50_values)
    best_map50_epoch = epochs_list[map50_values.index(best_map50)]

    print(f"\n{'='*60}")
    print(f"  CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Best mAP@0.5:        {best_map50:.4f} at epoch {best_map50_epoch}")
    if convergence_epoch:
        print(f"  Loss plateau reached: epoch ~{convergence_epoch}")
    else:
        print(f"  Loss trend: still improving at epoch {epochs_list[-1]}")

    # Compute improvement from epoch 1 to best
    first_map50 = map50_values[0]
    improvement = ((best_map50 - first_map50) / first_map50 * 100) if first_map50 > 0 else 0
    print(f"  Improvement:         {first_map50:.4f} → {best_map50:.4f} "
          f"(+{improvement:.1f}%)")


def generate_metrics_markdown(epochs, output_dir):
    """Generate a markdown table of metrics per epoch for the report."""
    md_path = output_dir / "metrics_summary.md"

    lines = ["# Training Metrics Summary", "", "| Epoch | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | LR | Time (s) |",
             "|-------|-----------|--------|----|---------|---------------|-----|----------|"]

    for e in epochs:
        try:
            epoch = e.get("epoch", "?")
            p = float(e.get("metrics/precision(B)", 0))
            r = float(e.get("metrics/recall(B)", 0))
            m50 = float(e.get("metrics/mAP50(B)", 0))
            m5095 = float(e.get("metrics/mAP50-95(B)", 0))
            lr = float(e.get("lr/pg0", 0))
            t = float(e.get("time", 0))
            f1 = f1_score(p, r)
            lines.append(f"| {epoch} | {p:.3f} | {r:.3f} | {f1:.3f} | {m50:.4f} | {m5095:.4f} | {lr:.6f} | {t:.1f} |")
        except (ValueError, KeyError):
            continue

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Metrics markdown: {md_path}")


def main():
    args = parse_args()

    # Find results.csv
    if args.results_csv:
        csv_path = Path(args.results_csv)
    elif args.run_dir:
        csv_path = find_results_csv(args.run_dir)
    else:
        # Try to find in common locations
        repo_root = Path(__file__).resolve().parent.parent
        for candidate in ["runs", "results"]:
            for run_dir in (repo_root / candidate).rglob("results.csv"):
                csv_path = run_dir
                break
            if csv_path.exists():
                break
        else:
            csv_path = repo_root / "results" / "results_v2.csv"

    if not csv_path or not csv_path.exists():
        print("Error: Could not find results.csv. Specify --results-csv or --run-dir.")
        sys.exit(1)

    print(f"  Reading: {csv_path}")
    epochs = parse_results_csv(csv_path)
    print(f"  Found {len(epochs)} epochs")

    output_dir = csv_path.parent
    if "runs" in str(csv_path):
        output_dir = csv_path.parent.parent / "results" / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Overall table
    best = compute_best_epoch(epochs)
    final = compute_final_epoch(epochs)
    generate_overall_table(best, final)

    # Convergence
    curve_data = generate_training_curves(epochs, output_dir)
    print_convergence_analysis(curve_data)

    # Markdown
    generate_metrics_markdown(epochs, output_dir)

    print("\n  Done.")


if __name__ == "__main__":
    main()