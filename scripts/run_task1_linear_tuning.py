from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_linear_tuning import run_linear_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务一非睡眠特征线性调参实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_linear_tuning(ROOT, overwrite=args.overwrite)
    print("任务一线性调参实验完成。")
    for path in written:
        print(f"- {path}")
