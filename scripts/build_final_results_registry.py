from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_results_registry import build_final_results_registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建复赛三任务最终机器可读结果注册表"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = build_final_results_registry(ROOT, overwrite=args.overwrite)
    print("最终机器可读结果注册表与跨文件审计完成。")
    print("本步骤未生成论文表格、图片或证据索引。")
    for path in written:
        print(f"- {path}")
