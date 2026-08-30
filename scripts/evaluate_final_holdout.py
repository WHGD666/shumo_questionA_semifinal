from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_holdout_evaluation import evaluate_final_holdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一次性评估三个冻结模型的内部最终留出集"
    )
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    written = evaluate_final_holdout(ROOT, args.confirm)
    print("三个任务的一次性内部最终留出集评估完成。")
    print("注意：以下结果不是组委会隐藏测试集官方成绩。")
    for path in written:
        print(f"- {path}")
