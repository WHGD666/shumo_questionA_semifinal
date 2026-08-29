from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_residual_correction import run_residual_correction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务二嵌套交叉拟合残差修正实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_residual_correction(ROOT, overwrite=args.overwrite)
    print("任务二嵌套交叉拟合残差修正实验完成。")
    for path in written:
        print(f"- {path}")
