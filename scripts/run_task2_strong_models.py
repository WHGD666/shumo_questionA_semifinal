from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_strong_models import run_strong_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务二周期时间编码下的固定强模型比较")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_strong_models(ROOT, overwrite=args.overwrite)
    print("任务二固定强模型比较完成。")
    for path in written:
        print(f"- {path}")
