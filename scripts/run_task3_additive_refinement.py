from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task3_additive_refinement import run_task3_additive_refinement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务三加性样条边界确认实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_task3_additive_refinement(ROOT, overwrite=args.overwrite)
    print("任务三加性样条边界确认完成。")
    for path in written:
        print(f"- {path}")
