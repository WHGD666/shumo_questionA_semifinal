from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_proxy_sensitivity import run_proxy_sensitivity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务一综合代理变量敏感性实验")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = run_proxy_sensitivity(ROOT, overwrite=args.overwrite)
    print("任务一代理变量敏感性实验完成。")
    for path in written:
        print(f"- {path}")
