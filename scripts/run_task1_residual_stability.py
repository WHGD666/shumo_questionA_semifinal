from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_residual_stability import run_stability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务一固定残差候选稳定性确认")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_stability(ROOT, overwrite=args.overwrite)
    print("任务一固定残差候选稳定性确认完成。")
    for path in written:
        print(f"- {path}")
