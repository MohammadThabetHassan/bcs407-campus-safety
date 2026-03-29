"""
BCS407 — Campus Safety Monitoring
Live Inference Script

Usage:
  python code/inference.py --source path/to/image.jpg
  python code/inference.py --source path/to/folder/
  python code/inference.py --source 0   # webcam
"""

import argparse
from ultralytics import YOLO
from pathlib import Path

def run_inference(source, conf=0.25, weights="model/weights/best.pt"):
    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        iou=0.45,
        save=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    print(f"\n📊 Detections:")
    print("-" * 50)
    classes = ['wet_floor_sign', 'fire_alarm', 'emergency_exit', 'safety_helmet']
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf   = float(box.conf)
            coords = box.xyxy[0].tolist()
            print(f"  ✅ {classes[cls_id]:<25} confidence: {conf:.2%}")
            print(f"     BBox: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")
        if len(r.boxes) == 0:
            print("  ⚠️  No objects detected")
    print(f"\nResults saved to: runs/detect/predict/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Campus Safety Object Detection")
    parser.add_argument("--source",  type=str, required=True, help="Image/folder/webcam source")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--weights", type=str, default="model/weights/best.pt")
    args = parser.parse_args()
    run_inference(args.source, args.conf, args.weights)
