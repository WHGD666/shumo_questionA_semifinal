from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_structured_models import run_structured_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务一结构化提升实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_structured_models(ROOT, overwrite=args.overwrite)
    print("任务一结构化提升实验完成。")
    for path in written:
        print(f"- {path}")
