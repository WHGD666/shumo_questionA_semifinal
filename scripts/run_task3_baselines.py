from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task3_baseline import run_task3_baselines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行复赛任务三双特征契约回归基线")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_task3_baselines(ROOT, overwrite=args.overwrite)
    print("任务三双特征契约回归基线完成。")
    for path in written:
        print(f"- {path}")
