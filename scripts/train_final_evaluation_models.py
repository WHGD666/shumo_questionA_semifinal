from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_evaluation_models import train_final_evaluation_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="仅使用固定8000条开发集训练复赛最终内部评估模型"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = train_final_evaluation_models(ROOT, overwrite=args.overwrite)
    print("三个内部评估模型训练完成；最终留出集仍未评估。")
    for path in written:
        print(f"- {path}")
