from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.task1_regression_baseline import run_task1

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
for path in run_task1(ROOT, overwrite=args.overwrite):
    print(f"- {path}")
