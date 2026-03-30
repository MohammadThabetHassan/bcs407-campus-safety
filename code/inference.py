"""
BCS407 — Campus Safety Monitoring
Live Inference Script

Usage:
  python code/inference.py --source path/to/image.jpg
  python code/inference.py --source path/to/folder/
  python code/inference.py --source 0   # webcam
  python code/inference.py --source path/to/image.jpg --weights model/weights/best_v2.pt
"""

import argparse
import sys
from ultralytics import YOLO
from pathlib import Path

CLASS_NAMES = ['wet_floor_sign', 'fire_alarm', 'emergency_exit', 'safety_helmet']

def run_inference(source, conf=0.25, weights="model/weights/best.pt"):
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"Error: weights file not found: {weights}")
        sys.exit(1)

    if source != "0" and not source.isdigit():
        source_path = Path(source)
        if not source_path.exists():
            print(f"Error: source not found: {source}")
            sys.exit(1)

    print(f"Loading model: {weights}")
    model = YOLO(weights)

    print(f"Running inference on: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        iou=0.45,
        save=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )

    total_detections = 0
    print("\nDetections:")
    print("-" * 50)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            box_conf = float(box.conf)
            coords = box.xyxy[0].tolist()
            if 0 <= cls_id < len(CLASS_NAMES):
                label = CLASS_NAMES[cls_id]
            else:
                label = f"class_{cls_id}"
            print(f"  {label:<25} confidence: {box_conf:.2%}")
            print(f"     BBox: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")
            total_detections += 1
        if len(r.boxes) == 0:
            print("  No objects detected")

    print(f"\nTotal detections: {total_detections}")
    print("Results saved to: runs/detect/predict/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Campus Safety Object Detection")
    parser.add_argument("--source",  type=str, required=True, help="Image/folder/webcam source (0 for webcam)")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--weights", type=str, default="model/weights/best.pt",
                        help="Path to model weights (use best_v2.pt for v2)")
    args = parser.parse_args()
    run_inference(args.source, args.conf, args.weights)
