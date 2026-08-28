from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.protocol_checks import validate_protocol

if __name__ == "__main__":
    candidates = [ROOT / "data" / "raw" / "A题数据集.csv", ROOT / "A题数据集.csv"]
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    print("复赛实验协议检查通过。")
    print(json.dumps(validate_protocol(data_path), ensure_ascii=False, indent=2))
