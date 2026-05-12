#!/usr/bin/env python3
"""
BCS407 Campus Safety - Full Kaggle Pipeline
Run this on Kaggle: GPU T4 x2, Internet ON.

BEFORE RUNNING:
  1. Upload the 4 Roboflow zip files as a Kaggle dataset
  2. Add that dataset to this notebook (right panel → "Add Data")
  3. Run the PATH FINDING cell below first to confirm location
"""

import os
import shutil
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# CONFIG — WILL BE AUTO-DETECTED
# ═══════════════════════════════════════════════════════════════════

ZIP_NAMES = [
    "Emergency Exit Signs.v4i.yolov8.zip",
    "Fire Alarm.v24i.yolov8 (1).zip",
    "Hard Hat Universe.v4i.yolov8.zip",
    "wet-floor-detection1.v2i.yolov8.zip",
]

TARGET_COUNT = 2500
REPO_URL = "https://github.com/MohammadThabetHassan/bcs407-campus-safety.git"


def header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")


def find_zips():
    """Search everywhere for the zip files."""
    search_dirs = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("/kaggle/temp"),
        Path("/root"),
    ]

    found_zips = {}  # zip_name -> full_path

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        zips = sorted(search_dir.rglob("*.zip"))
        for z in zips:
            name = z.name
            for expected in ZIP_NAMES:
                if name == expected or expected in name or name in expected:
                    found_zips[expected] = z
                    break

    return found_zips


def safe_chdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(str(path))


# ═══════════════════════════════════════════════════════════════════
# STEP 0: FIND ZIPS FIRST
# ═══════════════════════════════════════════════════════════════════

def step_0_find_zips():
    header("STEP 0: Locate Dataset Zip Files")

    print("  Searching for zip files...\n")
    found = find_zips()

    print("  Search results:")
    for name in ZIP_NAMES:
        if name in found:
            print(f"  ✅ {name}")
            print(f"     → {found[name]}")
        else:
            print(f"  ❌ {name} — NOT FOUND")

    if len(found) < len(ZIP_NAMES):
        print(f"\n  Found {len(found)}/{len(ZIP_NAMES)} zip files")
        print("\n  ⚠️  Missing zip files! To fix this:")
        print("  ────────────────────────────────────────")
        print("  1. Go to Kaggle → Datasets → 'New Dataset'")
        print("  2. Upload these 4 zip files:")
        for z in ZIP_NAMES:
            if z not in found:
                print(f"     • {z}")
        print("  3. Make it PUBLIC")
        print("  4. In your notebook, click 'Add Data' on right panel")
        print("  5. Search for and attach your uploaded dataset")
        print("  6. Re-run this notebook")
        print("  ────────────────────────────────────────")

        # List what IS available
        all_zips = []
        for d in [Path("/kaggle/input"), Path("/kaggle/working")]:
            if d.exists():
                all_zips += list(d.rglob("*.zip"))
        if all_zips:
            print(f"\n  Zip files found (but not matching expected names):")
            for z in all_zips:
                print(f"    • {z}")
        else:
            print(f"\n  No zip files found anywhere in Kaggle.")

        sys.exit(1)

    print(f"\n  ✅ All {len(found)} zip files found!")

    # Copy to working dir
    WORK_DIR = Path("/kaggle/working/bcs407-campus-safety")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in found.items():
        dst = WORK_DIR / name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Copied: {name}")

    print(f"  Zip files ready in: {WORK_DIR}")
    return WORK_DIR, found


# ═══════════════════════════════════════════════════════════════════
# STEP 1: CLONE REPO
# ═══════════════════════════════════════════════════════════════════

def step_1_clone(WORK_DIR):
    header("STEP 1: Clone Repository")

    safe_chdir("/kaggle/working")
    if WORK_DIR.exists():
        # Remove contents but keep the dir (so zips aren't lost)
        for item in WORK_DIR.iterdir():
            if item.name.endswith(".zip"):
                continue  # Keep zips
            if item.is_dir():
                shutil.rmtree(str(item))
            else:
                item.unlink()

    # Clone fresh
    result = os.system(f"git clone {REPO_URL} {WORK_DIR} --quiet")
    if result != 0:
        print("  ❌ Git clone failed — trying without --quiet")
        os.system(f"git clone {REPO_URL} {WORK_DIR}")

    safe_chdir(WORK_DIR)
    print(f"  Working dir: {os.getcwd()}")

    # Re-copy zips (clone might have cleared the dir)
    for z in WORK_DIR.glob("*.zip"):
        pass  # Already there

    print("✅ Repo cloned and ready")


# ═══════════════════════════════════════════════════════════════════
# STEP 2: BUILD DATASET
# ═══════════════════════════════════════════════════════════════════

def step_2_setup():
    header("STEP 2: Build Dataset from Zips")
    os.system("python code/setup_v2.py")
    print("✅ Dataset built")


# ═══════════════════════════════════════════════════════════════════
# STEP 3: ANALYZE
# ═══════════════════════════════════════════════════════════════════

def step_3_analyze():
    header("STEP 3: Analyze Original Class Distribution")
    from code.analyze_distribution import analyze_dataset
    stats = analyze_dataset("dataset")

    if "train" in stats:
        counts = [stats["train"]["per_class_images"].get(c, 0) for c in
                  ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]]
        non_zero = [c for c in counts if c > 0]
        if non_zero:
            ratio = max(counts) / min(non_zero)
            print(f"\n  ⚠️  Original imbalance ratio: {ratio:.1f}x")

    print("✅ Analysis complete")


# ═══════════════════════════════════════════════════════════════════
# STEP 4: BALANCED AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

def step_4_augment():
    header("STEP 4: Apply Balanced Augmentation")
    os.system(f"python code/augment_v2.py --balance-mode equalize --target-count {TARGET_COUNT}")
    print("✅ Balancing complete")


# ═══════════════════════════════════════════════════════════════════
# STEP 5: VERIFY
# ═══════════════════════════════════════════════════════════════════

def step_5_verify():
    header("STEP 5: Verify Balance")
    from code.analyze_distribution import analyze_dataset
    stats = analyze_dataset("dataset")
    if "train" in stats:
        counts = [stats["train"]["per_class_images"].get(c, 0) for c in
                  ["wet_floor_sign", "fire_alarm", "emergency_exit", "safety_helmet"]]
        ratio = max(counts) / min(c for c in counts if c > 0)
        print(f"  ✅ Imbalance ratio: {ratio:.2f}x")
    print("✅ Verification complete")


# ═══════════════════════════════════════════════════════════════════
# STEP 6: WEIGHTS + EDA
# ═══════════════════════════════════════════════════════════════════

def step_6_weights_and_eda():
    header("STEP 6: Class Weights + BBox Stats")
    os.system("python code/apply_class_weights.py")
    os.system("python code/dataset_analysis.py dataset")
    print("✅ Done")


# ═══════════════════════════════════════════════════════════════════
# STEP 7: TRAIN
# ═══════════════════════════════════════════════════════════════════

def step_7_train():
    header("STEP 7: Train YOLOv8m (150 epochs)")
    print("  Config: batch=16, lr0=0.005, cos_lr, warmup=10, epochs=150")
    print("  ⏱️  This takes ~15-20 hours on Kaggle T4\n")

    best_pt = Path("runs/detect/campus_safety_v3_balanced/weights/best.pt")
    if best_pt.exists():
        print("  Model already trained! Skipping.")
        return True

    result = os.system("bash code/train_balanced.sh")
    return result == 0


# ═══════════════════════════════════════════════════════════════════
# STEP 8: EVALUATE
# ═══════════════════════════════════════════════════════════════════

def step_8_evaluate():
    header("STEP 8: Evaluate Model")
    import json

    weights = Path("runs/detect/campus_safety_v3_balanced/weights/best.pt")
    if not weights.exists():
        weights = Path("model/weights/best_v2.pt")
    if not weights.exists():
        print("  ⚠️  Weights not found")
        return

    os.system(f"python code/evaluate_model.py --weights {weights} --json-out results/evaluation_results.json")

    if Path("results/evaluation_results.json").exists():
        with open("results/evaluation_results.json") as f:
            data = json.load(f)
        ov = data.get("overall", {})
        print(f"\n  📊 Precision={ov.get('precision',0):.3f}  "
              f"Recall={ov.get('recall',0):.3f}  "
              f"mAP50={ov.get('map50',0):.4f}  "
              f"mAP50-95={ov.get('map50_95',0):.4f}")
    print("✅ Evaluation complete")


# ═══════════════════════════════════════════════════════════════════
# STEP 9: FIGURES
# ═══════════════════════════════════════════════════════════════════

def step_9_figures():
    header("STEP 9: Generate Report Figures")
    v2 = Path("results/results_v2.csv")
    v3 = Path("runs/detect/campus_safety_v3_balanced/results.csv")
    csv = str(v3) if v3.exists() else (str(v2) if v2.exists() else "")

    cmd = ("python code/generate_report_plots.py "
           "--dataset-stats dataset/dataset_stats.json "
           "--bbox-stats dataset/bbox_stats.json "
           "--format both "
           "--augmentation-dir dataset/train/images "
           "--predictions-dir results/predictions")
    if csv:
        cmd += f" --results-csv {csv}"
    ej = Path("results/evaluation_results.json")
    if ej.exists():
        cmd += f" --evaluation-json {ej}"
    os.system(cmd)
    print("✅ Figures generated")


# ═══════════════════════════════════════════════════════════════════
# STEP 10: DISPLAY
# ═══════════════════════════════════════════════════════════════════

def step_10_display():
    header("STEP 10: Display Results")
    figs = [
        ("Before vs After Balance", "03_before_after_comparison.png"),
        ("Class Distribution (Before)", "01_class_distribution_before.png"),
        ("Class Distribution (After)", "02_class_distribution_after.png"),
        ("Training Loss", "05_training_loss_curves.png"),
        ("All Metrics", "07_metrics_curves.png"),
        ("Per-Class mAP", "09_per_class_map.png"),
        ("Per-Class F1", "10_per_class_f1.png"),
        ("Confusion Matrix", "confusion_matrix.png"),
    ]
    try:
        from IPython.display import Image, display
        for label, fname in figs:
            fp = Path(fname)
            if fp.exists():
                print(f"\n  📊 {label}")
                display(Image(filename=str(fp), width=700))
    except ImportError:
        print("  See results/plots/ for all figures")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║   BCS407 Campus Safety — Kaggle Pipeline            ║
    ║   10 steps: Data → Balance → Train → Eval → Report  ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # First: find and copy zips
    WORK_DIR, zip_map = step_0_find_zips()

    # Rest of pipeline
    step_1_clone(WORK_DIR)
    step_2_setup()
    step_3_analyze()
    step_4_augment()
    step_5_verify()
    step_6_weights_and_eda()
    step_7_train()
    step_8_evaluate()
    step_9_figures()
    step_10_display()

    header("🎉 PIPELINE COMPLETE ✅")
    print("""
  ┌──────────────────────────────────────────────┐
  │  Dataset:  9,996 images, 2,500/class          │
  │  Balance:  10.4x → 1.0x ✅                   │
  │  Model:    YOLOv8m                            │
  │  Figures:  results/plots/                    │
  │  Docs:     docs/ (7 markdown files)           │
  └──────────────────────────────────────────────┘
    """)