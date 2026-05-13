#!/usr/bin/env python3
"""
Model evaluation script — runs validation on test set, extracts per-class
precision/recall/F1, confusion matrix data, and inference speed.

Usage:
    python code/evaluate_model.py --weights model/weights/best_v3.pt
    python code/evaluate_model.py --weights runs/detect/campus_safety_v3_balanced/weights/best.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on test set.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to trained .pt weights file.",
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Path to data.yaml.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--json-out",
        default="results/evaluation_results.json",
        help="Where to save evaluation results JSON.",
    )
    return parser.parse_args()


def get_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)


def run_validation(weights_path, data_path, imgsz, batch, iou, conf):
    """Run YOLO validation and extract metrics."""
    YOLO = get_yolo()
    model = YOLO(weights_path)

    print(f"\n{'='*60}")
    print(f"  MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"  Weights:  {weights_path}")
    print(f"  Data:     {data_path}")
    print(f"  Imgsz:    {imgsz}")
    print(f"  Batch:    {batch}")
    print(f"  IoU:      {iou}")
    print(f"  Conf:     {conf}")
    print()

    # Run validation
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        batch=batch,
        iou=iou,
        conf=conf,
        device=0,
        verbose=True,
    )

    return results, model


def extract_detailed_metrics(results):
    """Extract per-class and overall metrics from YOLO val() results object.

    Ultralytics model.val() returns a DetMetrics object with:
      results.box.mp       — mean precision
      results.box.mr       — mean recall
      results.box.map50    — mAP@0.5
      results.box.map      — mAP@0.5:0.95
      results.box.f1       — F1 score (array, first element)
      results.box.p        — per-class precision array
      results.box.r        — per-class recall array
      results.box.ap50     — per-class mAP@0.5 array
      results.box.ap       — per-class mAP@0.5:0.95 array
      results.speed        — dict with preprocess/inference/postprocess ms
    """
    metrics = {
        "overall": {},
        "per_class": {},
        "speed": {},
    }

    # Overall metrics from results.box
    box = getattr(results, 'box', None)
    if box is not None:
        metrics["overall"]["precision"] = float(box.mp)
        metrics["overall"]["recall"] = float(box.mr)
        metrics["overall"]["map50"] = float(box.map50)
        metrics["overall"]["map50_95"] = float(box.map)
        # F1 may be an array or scalar
        f1_val = box.f1
        if hasattr(f1_val, '__len__'):
            metrics["overall"]["f1"] = float(f1_val[0]) if len(f1_val) > 0 else 0.0
        else:
            metrics["overall"]["f1"] = float(f1_val)

    # Speed
    if hasattr(results, 'speed'):
        metrics["speed"] = {
            "preprocess_ms": results.speed.get("preprocess", 0),
            "inference_ms": results.speed.get("inference", 0),
            "postprocess_ms": results.speed.get("postprocess", 0),
        }

    # Per-class metrics from results.box arrays
    if box is not None:
        try:
            p_arr = box.p    # per-class precision
            r_arr = box.r    # per-class recall
            ap50_arr = box.ap50  # per-class mAP@0.5
            ap_arr = box.ap  # per-class mAP@0.5:0.95
            for i, cls_name in enumerate(CLASS_NAMES):
                if i < len(p_arr):
                    p = float(p_arr[i])
                    r = float(r_arr[i])
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    metrics["per_class"][cls_name] = {
                        "precision": p,
                        "recall": r,
                        "f1": f1,
                        "map50": float(ap50_arr[i]),
                        "map50_95": float(ap_arr[i]),
                    }
        except Exception as e:
            print(f"  [INFO] Could not extract per-class metrics: {e}")

    return metrics


def run_inference_benchmark(weights_path, data_dir, num_images=200):
    """Benchmark inference speed on test images."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights_path)
    test_dir = Path(data_dir) / "test" / "images"
    if not test_dir.exists():
        print(f"  Warning: {test_dir} not found for benchmark")
        return None

    images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    images = images[:num_images]

    times = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        start = time.perf_counter()
        model.predict(img, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    if not times:
        return None

    return {
        "num_images": len(times),
        "mean_ms": round(sum(times) / len(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "median_ms": round(sorted(times)[len(times)//2], 2),
        "fps": round(1000 / (sum(times) / len(times)), 1),
    }


def save_results(metrics, benchmark, output_path):
    """Save evaluation results to JSON."""
    output = {
        "weights": str(metrics.get("weights_path", "")),
        "overall": metrics.get("overall", {}),
        "per_class": metrics.get("per_class", {}),
        "speed": metrics.get("speed", {}),
        "inference_benchmark": benchmark,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")


def print_report(metrics, benchmark):
    """Print formatted evaluation report."""
    print(f"\n{'='*80}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*80}")

    # Overall table
    ov = metrics["overall"]
    if ov:
        print(f"\n  Overall Metrics:")
        print(f"  {'Metric':<20} {'Value':>12}")
        print(f"  {'-'*35}")
        for key in ["precision", "recall", "f1", "map50", "map50_95"]:
            if key in ov:
                print(f"  {key:<20} {ov[key]:>10.4f}")

    # Per-class table
    if metrics["per_class"]:
        print(f"\n  Per-Class Metrics:")
        print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
        print(f"  {'-'*78}")
        for cls in CLASS_NAMES:
            if cls in metrics["per_class"]:
                c = metrics["per_class"][cls]
                print(f"  {cls:<20} {c.get('precision',0):>10.4f} {c.get('recall',0):>10.4f} "
                      f"{c.get('f1',0):>10.4f} {c.get('tp',0):>8} {c.get('fp',0):>8} {c.get('fn',0):>8}")

    # Speed
    if metrics["speed"]:
        print(f"\n  Inference Speed (model):")
        s = metrics["speed"]
        print(f"  {'Stage':<20} {'Time (ms)':>10}")
        print(f"  {'-'*32}")
        for stage, val in s.items():
            print(f"  {stage:<20} {val:>10.1f}")

    if benchmark:
        print(f"\n  Inference Benchmark ({benchmark['num_images']} test images):")
        print(f"  {'Metric':<20} {'Value':>12}")
        print(f"  {'-'*35}")
        for key in ["mean_ms", "min_ms", "max_ms", "median_ms", "fps"]:
            print(f"  {key:<20} {benchmark[key]:>10}")

    print(f"\n{'='*80}")


def main():
    args = parse_args()

    # Find weights if not specified
    weights = args.weights
    if not weights:
        repo = Path(__file__).resolve().parent.parent
        candidates = [
            "model/weights/best_v3.pt",
            "model/weights/best_v2.pt",
            "runs/detect/campus_safety_v3_balanced/weights/best.pt",
            "runs/detect/campus_safety_v2_balanced/weights/best.pt",
            "runs/detect/campus_safety_v2_fixed/weights/best.pt",
        ]
        for c in candidates:
            p = repo / c
            if p.exists():
                weights = str(p)
                break

    if not weights or not Path(weights).exists():
        print(f"Error: No weights file found. Provide --weights or train a model first.")
        sys.exit(1)

    print(f"\n  Using weights: {weights}")

    # Run validation
    results, model = run_validation(
        weights_path=weights,
        data_path=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        iou=args.iou,
        conf=args.conf,
    )

    # Extract metrics
    metrics = extract_detailed_metrics(results)
    metrics["weights_path"] = weights

    # Run benchmark
    print("\n  Running inference benchmark...")
    benchmark = run_inference_benchmark(
        weights_path=weights,
        data_dir=args.data.replace("data.yaml", ""),
        num_images=200,
    )

    # Print and save
    print_report(metrics, benchmark)

    output_path = Path(args.json_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(metrics, benchmark, output_path)

    print("\n  Evaluation complete.")


if __name__ == "__main__":
    main()