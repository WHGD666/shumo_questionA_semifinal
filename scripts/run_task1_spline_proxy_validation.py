from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_spline_proxy_validation import run_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务一最终样条代理变量验证")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_validation(ROOT, overwrite=args.overwrite)
    print("任务一最终样条代理变量验证完成。")
    for path in written:
        print(f"- {path}")
