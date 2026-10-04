from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import fetch_uci, load_local_excel, prepare_dataset
from .modeling import train_all

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe CardioMap models.")
    parser.add_argument("--data", type=str, default=None, help="Local .xlsx/.xls dataset path. Uses UCI 411 when omitted.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_MODEL_DIR), help="Model artifact directory.")
    args = parser.parse_args()

    df = load_local_excel(args.data) if args.data else fetch_uci()
    X, ys, target_columns = prepare_dataset(df)

    output_dir = Path(args.output)
    results = train_all(X, ys, output_dir)

    metadata = {
        "dataset_source": "UCI Extension of Z-Alizadeh Sani, dataset 411",
        "dataset_url": "https://archive.ics.uci.edu/dataset/411/extention%2Bof%2Bz%2B",
        "dataset_doi": "10.24432/C5461K",
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "target_columns": target_columns,
        "target_definitions": {
            "CAD": "Overall CAD status",
            "LAD": "LAD stenosis status",
            "LCX": "LCX stenosis status",
            "RCA": "RCA stenosis status",
        },
        "leakage_excluded_from_all_tasks": ["LAD", "LCX", "RCA", "Cath"],
        "clinical_note": "This is a research/hackathon prototype; metrics are not clinical validation.",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Trained {len(results)} models on {len(X)} rows and {X.shape[1]} leakage-safe features.")
    for target, fit in results.items():
        h = fit.metrics["holdout"]
        print(target, h)


if __name__ == "__main__":
    main()
