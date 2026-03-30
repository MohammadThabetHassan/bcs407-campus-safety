"""
BCS407 — Campus Safety Monitoring
Inference helper for image, folder, video, or webcam sources.
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


CLASS_NAMES = ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Campus Safety Object Detection")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, folder, video path, URL, or webcam index.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(repo_root / "model" / "weights" / "best.pt"),
        help="Path to the YOLO weights file.",
    )
    parser.add_argument(
        "--project", type=str, default="runs/detect", help="Output project directory."
    )
    parser.add_argument("--name", type=str, default="predict", help="Output run name.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, for example 0 or cpu.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display predictions live when supported."
    )
    parser.add_argument(
        "--nosave", action="store_true", help="Do not save rendered prediction outputs."
    )
    return parser.parse_args()


def validate_inputs(source: str, weights: str) -> Path:
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if source.isdigit():
        return weights_path

    if source.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return weights_path

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    return weights_path


def label_for_class(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def run_inference(args: argparse.Namespace) -> None:
    weights_path = validate_inputs(args.source, args.weights)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=0.45,
        save=not args.nosave,
        show=args.show,
        show_labels=True,
        show_conf=True,
        line_width=2,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    total_detections = 0
    save_dir = None
    print("\nDetections:")
    print("-" * 50)
    for result in results:
        save_dir = getattr(result, "save_dir", save_dir)
        if len(result.boxes) == 0:
            print("  No objects detected")
            continue

        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {label_for_class(class_id):<25} confidence: {confidence:.2%}")
            print(f"     BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            total_detections += 1

    print(f"\nTotal detections: {total_detections}")
    if save_dir and not args.nosave:
        print(f"Results saved to: {save_dir}")


def main() -> int:
    args = parse_args()
    try:
        run_inference(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
