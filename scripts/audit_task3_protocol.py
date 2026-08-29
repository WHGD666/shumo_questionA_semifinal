from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task3_protocol import audit_task3_protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="审计并冻结复赛任务三实验协议")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = audit_task3_protocol(ROOT, overwrite=args.overwrite)
    print("任务三实验协议审计完成。")
    for path in written:
        print(f"- {path}")
