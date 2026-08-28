from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_spline_tuning import run_spline_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务一样条 Ridge 有界调参实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_spline_tuning(ROOT, overwrite=args.overwrite)
    print("任务一样条 Ridge 调参实验完成。")
    for path in written:
        print(f"- {path}")
