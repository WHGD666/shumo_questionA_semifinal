from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 2026

if __name__ == "__main__":
    candidates = [ROOT / "data" / "raw" / "A题数据集.csv", ROOT / "A题数据集.csv"]
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("未找到复赛数据集")
    frame = pd.read_csv(data_path)
    ids = frame["Person_ID"].astype(str)
    development, holdout = train_test_split(ids, test_size=0.20, random_state=SEED, shuffle=True)
    assignments = pd.DataFrame({"Person_ID": ids, "split": "development", "cv_fold": -1})
    assignments.loc[assignments["Person_ID"].isin(set(holdout)), "split"] = "holdout"
    dev_ids = assignments.loc[assignments["split"] == "development", "Person_ID"].to_numpy()
    for fold, (_, valid_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=SEED).split(dev_ids)):
        valid_ids = set(dev_ids[valid_idx])
        assignments.loc[assignments["Person_ID"].isin(valid_ids), "cv_fold"] = fold
    out_dir = ROOT / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out_dir / "split_assignments.csv", index=False, encoding="utf-8-sig")
    manifest = {"seed": SEED, "development_fraction": 0.8, "holdout_fraction": 0.2, "cv_folds": 5, "rows": len(frame), "data_file": data_path.name}
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("复赛固定数据划分完成。")
    print(f"- {out_dir / 'split_assignments.csv'}")
    print(f"- {out_dir / 'split_manifest.json'}")
