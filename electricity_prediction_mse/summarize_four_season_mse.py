# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


EXP_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = EXP_DIR / "experiment_summary.json"
OUTPUT_PATH = EXP_DIR / "plots" / "four_season_mse_mean_variance.csv"

SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
METHODS = [
    ("Missing Data", "Prediction w/o GAIN"),
    ("GAIN Imputed Data", "Prediction with GAIN"),
    ("Guided GAIN Data", "Prediction with fine-tuned GAIN"),
    ("Original Data", "Prediction with ture data"),
]
MISSING_RATES = ["0.1", "0.2", "0.3", "0.4", "0.5"]


def main() -> None:
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    mse_results = summary["mse_results"]
    rows = []
    for rate_index, rate in enumerate(MISSING_RATES):
        for method_key, method_label in METHODS:
            values = np.array(
                [mse_results[season][method_key][rate_index] for season in SEASONS],
                dtype=float,
            )
            rows.append(
                {
                    "missing_rate": rate,
                    "method": method_label,
                    "winter_mse": values[0],
                    "spring_mse": values[1],
                    "summer_mse": values[2],
                    "autumn_mse": values[3],
                    "mean_mse": float(np.mean(values)),
                    "variance_mse": float(np.var(values, ddof=0)),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved four-season MSE summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
