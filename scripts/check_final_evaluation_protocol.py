from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_evaluation_protocol import write_protocol_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查复赛三个任务的最终评估协议与留出集门禁")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = write_protocol_audit(ROOT, overwrite=args.overwrite)
    print("复赛最终评估协议检查通过；留出集仍保持封存。")
    for path in written:
        print(f"- {path}")
