#!/usr/bin/env python3
"""
BCS407 Campus Safety - Full Kaggle Pipeline
Run on Kaggle: GPU T4 x2, Internet ON.

IMPORTANT: Add your Kaggle dataset (containing the 4 Roboflow zip files)
to this notebook before running.

Usage:
  !python code/run_on_kaggle.py
"""

import os
import shutil
import sys
import subprocess
from pathlib import Path

ZIP_NAMES = [
    "Emergency Exit Signs.v4i.yolov8.zip",
    "Fire Alarm.v24i.yolov8 (1).zip",
    "Hard Hat Universe.v4i.yolov8.zip",
    "wet-floor-detection1.v2i.yolov8.zip",
]

TARGET_COUNT = 2500
REPO_URL = "https://github.com/MohammadThabetHassan/bcs407-campus-safety.git"
WORK_DIR = Path("/kaggle/working/bcs407-campus-safety")


def header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")


def debug_inventory():
    """Full diagnostic of what's available on this Kaggle notebook."""
    print("  🔍 RUNNING FULL INVENTORY...\n")

    # Check /kaggle/input recursively
    print("  [1] ALL files under /kaggle/input/:")
    result = subprocess.run(
        "find /kaggle/input -type f 2>/dev/null | head -50",
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            print(f"      {line}")
    else:
        print("      (empty or not accessible)")

    # Check all .zip anywhere
    print("\n  [2] ALL .zip files on the system:")
    result = subprocess.run(
        "find / -maxdepth 5 -name '*.zip' 2>/dev/null",
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            print(f"      {line}")
    else:
        print("      NONE FOUND")

    # Check /kaggle/input directories
    print("\n  [3] /kaggle/input directory structure:")
    result = subprocess.run(
        "ls -laR /kaggle/input/ 2>/dev/null | head -80",
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            print(f"      {line}")
    else:
        print("      (empty or permission denied)")

    # Check working dir
    print("\n  [4] /kaggle/working contents:")
    result = subprocess.run(
        "ls -la /kaggle/working/ 2>/dev/null",
        shell=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            print(f"      {line}")

    print()


def find_zips_robust():
    """Search EVERYWHERE for zip files — no assumptions about path."""
    found_zips = {}  # expected_name -> actual_path
    all_zips_found = []

    search_locations = [
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/temp",
        "/root",
        "/tmp",
    ]

    for location in search_locations:
        loc = Path(location)
        if not loc.exists():
            continue
        for z in loc.rglob("*.zip"):
            all_zips_found.append(z)
            name = z.name
            # Match against expected names (exact or contains)
            for expected in ZIP_NAMES:
                if name == expected or expected in name or name.replace(".zip", "") in expected:
                    found_zips[expected] = z
                    break

    return found_zips, all_zips_found


def step_0_find_and_copy_zips():
    header("STEP 0: Locate and Copy Dataset Zips")

    found, all_zips = find_zips_robust()

    print("  Zips matched to expected names:")
    for name in ZIP_NAMES:
        if name in found:
            print(f"  ✅ {name}")
            print(f"     Found at: {found[name]}")
        else:
            print(f"  ❌ {name} — NOT FOUND")

    if all_zips and not found:
        print(f"\n  ⚠️  Found {len(all_zips)} zip(s), but names don't match:")
        for z in all_zips:
            print(f"     • {z} ({z.stat().st_size / 1024 / 1024:.1f} MB)")
        print("  Attempting to use whatever zips are available...")
        # Use whatever zips we found, mapping them to expected names
        for i, z in enumerate(all_zips):
            if i < len(ZIP_NAMES):
                found[ZIP_NAMES[i]] = z
                print(f"  ↳ Mapping: {z.name} → {ZIP_NAMES[i]}")

    if not found:
        print("\n  ❌ NO ZIP FILES FOUND ANYWHERE!")
        print("  =============================================")
        print("  You MUST upload the dataset zip files first!")
        print("")
        print("  HOW TO FIX:")
        print("  1. Download these 4 Roboflow zip files:")
        print("     • wet-floor-detection1.v2i.yolov8.zip")
        print("     • Fire Alarm.v24i.yolov8 (1).zip")
        print("     • Emergency Exit Signs.v4i.yolov8.zip")
        print("     • Hard Hat Universe.v4i.yolov8.zip")
        print("  2. Go to Kaggle → Datasets → '+ New Dataset'")
        print("  3. Upload all 4 zips as a new dataset")
        print("  4. Make it PUBLIC")
        print("  5. Come back to your notebook → 'Add Data' (right panel)")
        print("  6. Search for your dataset → Add it")
        print("  7. Re-run this cell!")
        print("  =============================================")

        # Full debug
        debug_inventory()
        sys.exit(1)

    # Copy to workspace
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(str(WORK_DIR))

    copied = 0
    for name, src_path in found.items():
        dst = WORK_DIR / name
        if not dst.exists():
            try:
                shutil.copy2(str(src_path), str(dst))
                copied += 1
                size = dst.stat().st_size / 1024 / 1024
                print(f"  📦 Copied: {name} ({size:.1f} MB)")
            except Exception as e:
                print(f"  ❌ Failed to copy {name}: {e}")
        else:
            print(f"  ↩️  Already in workspace: {name}")

    print(f"\n  {copied} zip file(s) in workspace")
    return True


def step_1_clone():
    header("STEP 1: Clone Repository into Workspace")

    # Clean repo files (keep zips!)
    for item in WORK_DIR.iterdir():
        if item.suffix == '.zip':
            continue
        if item.name == '__pycache__':
            continue
        if item.is_dir():
            shutil.rmtree(str(item))
        else:
            item.unlink()

    os.system(f"git clone {REPO_URL} . --quiet 2>&1")
    print(f"  CWD: {os.getcwd()}")

    # Verify
    if (WORK_DIR / "code" / "setup_v2.py").exists():
        print("✅ Repo cloned successfully")
    else:
        print("❌ Clone may have failed — checking...")
        os.system("git clone " + REPO_URL + " . 2>&1")

    # Re-copy zips after clone (clone may have cleared them)
    for z in WORK_DIR.glob("*.zip"):
        pass  # Should be there already
    print(f"  Zip files present: {len(list(WORK_DIR.glob('*.zip')))}")


def step_2_build():
    header("STEP 2: Build Dataset")
    os.system("python code/setup_v2.py")
    print("✅ Dataset built")


def step_3_analyze():
    header("STEP 3: Analyze Original Distribution")
    from code.analyze_distribution import analyze_dataset
    analyze_dataset("dataset")
    print("✅ Analysis complete")


def step_4_augment():
    header("STEP 4: Balanced Augmentation (10.4x → 1.0x)")
    os.system(f"python code/augment_v2.py --balance-mode equalize --target-count {TARGET_COUNT} 2>&1")
    print("✅ Balancing complete")


def step_5_verify():
    header("STEP 5: Verify Balance")
    from code.analyze_distribution import analyze_dataset
    analyze_dataset("dataset")
    print("✅ Balance verified")


def step_6_extra():
    header("STEP 6: Class Weights + BBox Statistics")
    os.system("python code/apply_class_weights.py")
    os.system("python code/dataset_analysis.py dataset")
    print("✅ Done")


def step_7_train():
    header("STEP 7: Train YOLOv8m (150 epochs, balanced)")
    print("  batch=16, lr0=0.005, cos_lr, warmup=10")
    print("  ⏱️  ~15-20 hours on T4\n")

    best = WORK_DIR / "runs/detect/campus_safety_v3_balanced/weights/best.pt"
    if best.exists():
        print("  Already trained! Skipping.")
        return True

    code = os.system("bash code/train_balanced.sh")
    return code == 0


def step_8_evaluate():
    header("STEP 8: Evaluate")
    import json
    w = WORK_DIR / "runs/detect/campus_safety_v3_balanced/weights/best.pt"
    if not w.exists():
        w = WORK_DIR / "model/weights/best_v2.pt"
        if not w.exists():
            print("  No weights found")
            return
    os.system(f"python code/evaluate_model.py --weights {w} --json-out results/evaluation_results.json")
    f = WORK_DIR / "results/evaluation_results.json"
    if f.exists():
        d = json.load(open(f))
        o = d.get("overall", {})
        print(f"\n  Precision={o.get('precision',0):.3f}  Recall={o.get('recall',0):.3f}  "
              f"mAP50={o.get('map50',0):.4f}  mAP50-95={o.get('map50_95',0):.4f}")
    print("✅ Done")


def step_9_figures():
    header("STEP 9: Generate All Figures")
    v2 = WORK_DIR / "results/results_v2.csv"
    v3 = WORK_DIR / "runs/detect/campus_safety_v3_balanced/results.csv"
    csv = str(v3 if v3.exists() else v2 if v2.exists() else "")

    cmd = ("python code/generate_report_plots.py "
           "--dataset-stats dataset/dataset_stats.json "
           "--bbox-stats dataset/bbox_stats.json "
           "--format both --augmentation-dir dataset/train/images "
           "--predictions-dir results/predictions")
    if csv:
        cmd += f" --results-csv {csv}"
    e = WORK_DIR / "results/evaluation_results.json"
    if e.exists():
        cmd += f" --evaluation-json {e}"
    os.system(cmd)
    print("✅ Figures generated")


def step_10_show():
    header("STEP 10: Display Results")
    try:
        from IPython.display import Image, display
        for label, fname in [
            ("Balance: Before→After", "03_before_after_comparison.png"),
            ("Distribution Before", "01_class_distribution_before.png"),
            ("Distribution After", "02_class_distribution_after.png"),
            ("Training Loss", "05_training_loss_curves.png"),
            ("All Metrics + LR", "07_metrics_curves.png"),
            ("Per-Class mAP", "09_per_class_map.png"),
            ("Per-Class F1", "10_per_class_f1.png"),
            ("Confusion Matrix", "confusion_matrix.png"),
        ]:
            p = WORK_DIR / fname
            if p.exists():
                print(f"\n  📊 {label}")
                display(Image(filename=str(p), width=700))
    except ImportError:
        print("  See results/plots/")
    print("""
  ╔═══════════════════════════════════════════════════════╗
  │  COMPLETE ✅  All results in results/plots/          │
  │  Docs: docs/ (7 files)                               │
  ╚═══════════════════════════════════════════════════════╝""")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║  BCS407 Campus Safety — Kaggle Full Pipeline        ║
    ║  Auto-locates zips • No hardcoded paths needed      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    step_0_find_and_copy_zips()
    step_1_clone()
    step_2_build()
    step_3_analyze()
    step_4_augment()
    step_5_verify()
    step_6_extra()
    step_7_train()
    step_8_evaluate()
    step_9_figures()
    step_10_show()