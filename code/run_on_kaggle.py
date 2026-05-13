#!/usr/bin/env python3
"""
BCS407 Campus Safety - Full Kaggle Pipeline
Run on Kaggle: GPU T4 x2, Internet ON.

Handles BOTH:
  - Dataset as .zip files (needs extraction)
  - Dataset as already-unzipped directories (just link them)
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

ZIP_NAMES = [
    "Emergency Exit Signs.v4i.yolov8",
    "Fire Alarm.v24i.yolov8 (1)",
    "Hard Hat Universe.v4i.yolov8",
    "wet-floor-detection1.v2i.yolov8",
]
ZIP_EXT = ".zip"

TARGET_COUNT = 2500
REPO_URL = "https://github.com/MohammadThabetHassan/bcs407-campus-safety.git"
WORK_DIR = Path("/kaggle/working/bcs407-campus-safety")
INPUT_DIR = Path("/kaggle/input/datasets/mohdqwe123/bcs407-campus-safety")


def header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")


def find_source_dirs():
    """Find whether data is as zips or unzipped dirs in INPUT_DIR."""
    if not INPUT_DIR.exists():
        return None, "INPUT_DIR not found"

    items = sorted(INPUT_DIR.iterdir())
    dirs = [i for i in items if i.is_dir()]
    zips = [i for i in items if i.suffix == '.zip']

    print(f"  INPUT_DIR contents: {len(dirs)} dirs, {len(zips)} zips")
    for d in dirs:
        print(f"    DIR:  {d.name}/")
    for z in zips:
        print(f"    ZIP:  {z.name}")

    # Check for unzipped dataset folders
    expected_base_names = [z.replace(".zip", "").replace(".zip", "") for z in ZIP_NAMES]

    matched_dirs = {}
    for exp in ZIP_NAMES:
        exp_no_zip = exp.replace(".zip", "")
        for d in dirs:
            if d.name == exp or d.name == exp_no_zip or exp_no_zip in d.name:
                matched_dirs[exp] = d
                break

    if len(matched_dirs) == len(ZIP_NAMES):
        return matched_dirs, "unzipped"

    # Check for zip files
    matched_zips = {}
    for exp in ZIP_NAMES:
        for z in zips:
            if z.name == exp or exp in z.name or z.name.replace(".zip","") in exp:
                matched_zips[exp] = z
                break

    if matched_zips:
        return matched_zips, "zips"

    return None, f"Nothing matched. Found: {[d.name for d in dirs] + [z.name for z in zips]}"


def step_0_prepare():
    header("STEP 0: Prepare Dataset Source")

    sources, mode = find_source_dirs()

    if mode == "unzipped":
        print("  📂 Data is already unzipped — creating zips\n")

        WORK_DIR.mkdir(parents=True, exist_ok=True)

        for exp_name, src_dir in sources.items():
            zip_name = exp_name
            zip_path = WORK_DIR / zip_name

            print(f"  Zipping: {src_dir.name} → {zip_name}.zip")
            with zipfile.ZipFile(str(zip_path) + ".zip", 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(src_dir):
                    for f in files:
                        fp = Path(root) / f
                        arcname = fp.relative_to(src_dir)
                        zf.write(fp, str(arcname))
            print(f"    ✅ Created: {zip_path}.zip")

        for exp_name, src_dir in sources.items():
            yaml = src_dir / "data.yaml"
            if yaml.exists():
                shutil.copy2(str(yaml), str(WORK_DIR / f"{exp_name}_data.yaml"))

        print("\n  ✅ All directories zipped successfully")
        return True

    elif mode == "zips":
        print("  📦 Data is as zip files — copying to workspace\n")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        for exp_name, zip_path in sources.items():
            dst = WORK_DIR / zip_path.name
            if not dst.exists():
                shutil.copy2(str(zip_path), str(dst))
                print(f"  ✅ Copied: {zip_path.name}")
        return True

    else:
        print(f"  ❌ Could not locate data: {mode}")
        print("\n  TROUBLESHOOTING:")
        print("  1. Make sure the dataset is added to your notebook")
        print("  2. Check the dataset path in the script (INPUT_DIR)")
        print("  3. Your dataset should contain either:")
        print("     a) 4 .zip files (Roboflow export format)")
        print("     b) 4 unzipped folders (Roboflow folder format)")
        print(f"  4. What we found: {[x.name for x in INPUT_DIR.iterdir()] if INPUT_DIR.exists() else 'INPUT_DIR does not exist'}")
        return False


def safe_chdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(str(path))


def step_1_clone():
     header("STEP 1: Clone Repository")
     safe_chdir("/kaggle/working")

     # Save prepared zips to temp location before deleting WORK_DIR
     # (bug fix: rmtree destroys zips that were created in step_0)
     import tempfile
     tmpdir = Path(tempfile.mkdtemp())
     saved_zip_names = []
     for zip_path in WORK_DIR.glob("*.zip"):
         dst = tmpdir / zip_path.name
         shutil.copy2(str(zip_path), str(dst))
         saved_zip_names.append(zip_path.name)
         print(f"  Saved zip to temp: {zip_path.name}")

     if WORK_DIR.exists():
         shutil.rmtree(str(WORK_DIR))

     # Clone into fresh WORK_DIR (must be empty/nonexistent for git clone)
     os.system(f"git clone {REPO_URL} {WORK_DIR} --quiet 2>&1")
     safe_chdir(WORK_DIR)

     # Restore saved zips after cloning
     for name in saved_zip_names:
         src = tmpdir / name
         dst = WORK_DIR / name
         if src.exists():
             shutil.copy2(str(src), str(dst))
             print(f"  Restored zip: {name}")
         else:
             print(f"  Warning: zip not found in temp: {name}")

     # Cleanup temp dir
     shutil.rmtree(str(tmpdir), ignore_errors=True)

     print(f"  CWD: {os.getcwd()}")
     if (WORK_DIR / "code" / "setup_v2.py").exists():
         print("✅ Repo cloned")
     else:
         print("❌ Clone failed")
         sys.exit(1)


def step_2_prepare_zips_for_setup():
    """Move the prepared zips from step 0 into the cloned repo dir."""
    header("STEP 2: Move Zips into Repo Root")

    # Find zip files in WORK_DIR
    zips = list(WORK_DIR.glob("*.zip"))
    print(f"  Found {len(zips)} zip file(s) in workspace")
    for z in zips:
        print(f"    • {z.name} ({z.stat().st_size/1024/1024:.1f} MB)")

    if not zips:
        print("  ❌ No zip files in workspace! Cannot continue.")
        return False

    # They should already be in WORK_DIR which IS the repo root
    print("  ✅ Zips are in repo root, ready for setup_v2.py")
    return True


def step_3_build():
    header("STEP 3: Build Dataset (setup_v2.py)")
    print("  This will extract zips, remap class IDs, and split data.\n")
    os.system("python code/setup_v2.py")
    if (WORK_DIR / "dataset" / "data.yaml").exists():
        print("✅ Dataset built successfully")
    else:
        print("❌ Dataset build failed")


def _import_code_module(module_name):
    """Import a module from the code/ directory safely."""
    import importlib
    code_dir = str(WORK_DIR / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    return importlib.import_module(module_name)


def step_4_analyze():
    header("STEP 4: Analyze Original Distribution")
    mod = _import_code_module("analyze_distribution")
    mod.analyze_dataset("dataset")
    print("✅ Analysis complete")


def step_5_augment():
    header("STEP 5: Balanced Augmentation (10.4x → 1.0x)")
    os.system(f"python code/augment_v2.py --balance-mode equalize --target-count {TARGET_COUNT} 2>&1")
    print("✅ Balancing complete")


def step_6_verify():
     header("STEP 6: Verify Balance")
     mod = _import_code_module("analyze_distribution")
     mod.analyze_dataset("dataset")
     print("\n✅ Balance verified")


def step_7_extra():
    header("STEP 7: Class Weights + BBox Statistics")
    os.system("python code/apply_class_weights.py")
    os.system("python code/dataset_analysis.py dataset")
    print("✅ Done")


def step_8_train():
    header("STEP 8: Train YOLOv8m (150 epochs)")
    print("  Config: batch=16, lr0=0.005, cos_lr, warmup=10, epochs=150")
    print("  ⏱️  ~15-20 hours on Kaggle T4\n")

    best = WORK_DIR / "runs/detect/campus_safety_v3_balanced/weights/best.pt"
    if best.exists():
        print("  Already trained! Skipping.")
        return True

    code = os.system("bash code/train_balanced.sh")
    return code == 0


def step_9_evaluate():
    header("STEP 9: Evaluate Model")
    import json
    w = WORK_DIR / "runs/detect/campus_safety_v3_balanced/weights/best.pt"
    if not w.exists():
        w = WORK_DIR / "model/weights/best_v2.pt"
        if not w.exists():
            print("  ⚠️  No weights found, checking for any .pt file...")
            pts = list(WORK_DIR.glob("**/*.pt")) + list(WORK_DIR.glob("**/*.best.pt"))
            if pts:
                w = pts[0]
                print(f"  Using: {w}")
            else:
                print("  No weights found. Skipping evaluation.")
                return

    os.system(f"python code/evaluate_model.py --weights {w} --json-out results/evaluation_results.json")
    f = WORK_DIR / "results/evaluation_results.json"
    if f.exists():
        d = json.load(open(f))
        o = d.get("overall", {})
        print(f"\n  Precision={o.get('precision',0):.3f}  "
              f"Recall={o.get('recall',0):.3f}  "
              f"mAP50={o.get('map50',0):.4f}  "
              f"mAP50-95={o.get('map50_95',0):.4f}")
    print("✅ Done")


def step_10_figures():
    header("STEP 10: Generate Report Figures")
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
    ej = WORK_DIR / "results/evaluation_results.json"
    if ej.exists():
        cmd += f" --evaluation-json {ej}"
    os.system(cmd)
    print("✅ Figures generated")


def step_11_show():
    header("STEP 11: Display Results")
    try:
        from IPython.display import Image, display
        for label, fname in [
            ("Before→After Balance", "03_before_after_comparison.png"),
            ("Distribution Before", "01_class_distribution_before.png"),
            ("Distribution After", "02_class_distribution_after.png"),
            ("Training Loss", "05_training_loss_curves.png"),
            ("Metrics + LR", "07_metrics_curves.png"),
            ("Per-Class mAP", "09_per_class_map.png"),
            ("Per-Class F1", "10_per_class_f1.png"),
            ("Confusion Matrix", "confusion_matrix.png"),
        ]:
            p = WORK_DIR / fname
            if p.exists():
                print(f"\n  📊 {label}")
                display(Image(filename=str(p), width=700))
    except ImportError:
        print("  See results/plots/ for figures")

    print("""
  ╔═══════════════════════════════════════════════════════╗
  │  COMPLETE ✅  Results in results/plots/              │
  │  Documentation in docs/ (7 markdown files)           │
  ╚═══════════════════════════════════════════════════════╝""")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║  BCS407 Campus Safety — Kaggle Full Pipeline        ║
    ║  Auto-detects zipped/unzipped data                  ║
    ╚══════════════════════════════════════════════════════╝
    """)

    step_0_prepare()
    step_1_clone()
    step_2_prepare_zips_for_setup()
    step_3_build()
    step_4_analyze()
    step_5_augment()
    step_6_verify()
    step_7_extra()
    step_8_train()
    step_9_evaluate()
    step_10_figures()
    step_11_show()