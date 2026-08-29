from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task3_linear_tuning import run_task3_linear_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务三双契约线性模型有界调参")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_task3_linear_tuning(ROOT, overwrite=args.overwrite)
    print("任务三线性模型有界调参完成。")
    for path in written:
        print(f"- {path}")
