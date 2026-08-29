from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_time_encoding import run_time_encoding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务二周期时间特征编码实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_time_encoding(ROOT, overwrite=args.overwrite)
    print("任务二周期时间特征编码实验完成。")
    for path in written:
        print(f"- {path}")
