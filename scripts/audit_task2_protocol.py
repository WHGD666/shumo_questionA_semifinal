from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_protocol import audit_task2_protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务二特征、代理变量与多重共线性协议审查")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = audit_task2_protocol(ROOT, overwrite=args.overwrite)
    print("任务二协议与共线性审查完成。")
    for path in written:
        print(f"- {path}")
