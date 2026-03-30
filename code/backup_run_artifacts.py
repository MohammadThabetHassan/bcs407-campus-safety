import argparse
import shutil
from pathlib import Path


ARTIFACTS = [
    "args.yaml",
    "results.csv",
    "weights/best.pt",
    "weights/last.pt",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Copy the minimum run artifacts needed to inspect or resume training."
    )
    parser.add_argument(
        "--run-dir",
        default=str(repo_root / "runs" / "detect" / "campus_safety_v2_fixed"),
        help="YOLO run directory containing args.yaml, results.csv, and weights/.",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory where the selected artifacts will be copied.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    dest_dir = Path(args.dest).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for relative_path in ARTIFACTS:
        source = run_dir / relative_path
        if not source.exists():
            continue
        destination = dest_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    if not copied:
        raise FileNotFoundError(
            f"No backup artifacts found in {run_dir}. Expected one of: {', '.join(ARTIFACTS)}"
        )

    print("Copied artifacts:")
    for path in copied:
        print(path)


if __name__ == "__main__":
    main()
