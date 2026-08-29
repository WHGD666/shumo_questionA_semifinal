from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_stage_report import build_task2_stage_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成任务二阶段报告与论文证据索引")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = build_task2_stage_report(ROOT, overwrite=args.overwrite)
    print("任务二阶段报告与论文证据索引生成完成。")
    for path in written:
        print(f"- {path}")
