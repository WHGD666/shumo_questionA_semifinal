from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submission_package import build_submission_package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建复赛A题统一提交包")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = build_submission_package(ROOT, overwrite=args.overwrite)
    print("统一提交包构建完成。")
    for path in paths:
        print(f"- {path}")
