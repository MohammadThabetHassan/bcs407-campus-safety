#!/usr/bin/env python3
"""
Balanced Training Script (v3)
Uses ultralytics Python API directly — no CLI dependency.
"""

from ultralytics import YOLO

def main():
    print("=== Balanced Training Config (v3) ===")
    print("Model:       YOLOv8m")
    print("Epochs:      150")
    print("LR0:         0.005")
    print("LRF:         0.01")
    print("Warmup:      10 epochs")
    print("Batch:       16")
    print("Imgsz:       640")
    print("")

    model = YOLO("yolov8m.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache=False,
        cos_lr=True,
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=10,
        momentum=0.937,
        weight_decay=0.0005,
        project="runs",
        name="campus_safety_v3_balanced",
        exist_ok=True,
    )

    print("\n✅ Training complete!")
    print(f"  Best weights: runs/detect/campus_safety_v3_balanced/weights/best.pt")
    return results

if __name__ == "__main__":
    main()
