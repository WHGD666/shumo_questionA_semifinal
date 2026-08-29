from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_elastic_tuning import run_elastic_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务二周期时间编码下的 Elastic Net 有界调参")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_elastic_tuning(ROOT, overwrite=args.overwrite)
    print("任务二 Elastic Net 有界调参实验完成。")
    for path in written:
        print(f"- {path}")
