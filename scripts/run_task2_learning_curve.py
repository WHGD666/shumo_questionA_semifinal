from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_learning_curve import run_learning_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务二冻结 Elastic Net 学习曲线诊断")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_learning_curve(ROOT, overwrite=args.overwrite)
    print("任务二冻结 Elastic Net 学习曲线诊断完成。")
    for path in written:
        print(f"- {path}")
