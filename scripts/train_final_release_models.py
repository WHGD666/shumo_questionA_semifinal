from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_release import train_final_release_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练复赛三任务最终发布模型")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = train_final_release_models(ROOT, overwrite=args.overwrite)
    print("三任务最终发布模型训练完成。")
    print("模型结构和参数保持冻结；本步骤未继续调参。")
    for path in paths:
        print(f"- {path}")
