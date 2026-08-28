from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_audit import run_audit


if __name__ == "__main__":
    paths = run_audit(ROOT)
    print("复赛数据审计完成。")
    for path in paths:
        print(f"- {path}")
