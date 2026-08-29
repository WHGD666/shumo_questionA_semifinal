from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task3_additive_models import run_task3_additive_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务三双契约定向加性样条实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_task3_additive_models(ROOT, overwrite=args.overwrite)
    print("任务三定向加性样条实验完成。")
    for path in written:
        print(f"- {path}")
