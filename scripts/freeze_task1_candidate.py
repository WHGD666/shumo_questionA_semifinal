from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_candidate_freeze import freeze_task1_candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冻结任务一开发集候选并生成实验登记表")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = freeze_task1_candidate(ROOT, overwrite=args.overwrite)
    print("任务一开发集候选冻结检查完成。")
    for path in written:
        print(f"- {path}")
