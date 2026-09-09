# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from build_2014_experiment import Predictor2014


EXP_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = EXP_DIR / "experiment_summary.json"
OUTPUT_PATH = EXP_DIR / "plots" / "four_season_all_points_mse_variance.csv"
SEASON_OUTPUT_PATH = EXP_DIR / "plots" / "season_mse_variance.csv"
SEASON_FALLBACK_OUTPUT_PATH = EXP_DIR / "plots" / "season_mse_variance_full.csv"

MONTH_ORDER = ["January", "April", "July", "November"]
SEASON_BY_MONTH = {
    "January": "Winter",
    "April": "Spring",
    "July": "Summer",
    "November": "Autumn",
}
RATES = ["0.1", "0.2", "0.3", "0.4", "0.5"]
METHOD_CONFIGS = [
    ("Missing Data", "Prediction w/o GAIN", "missing"),
    ("GAIN Imputed Data", "Prediction with GAIN", "gain_imputed"),
    ("Guided GAIN Data", "Prediction with fine-tuned GAIN", "guided_gain_imputed"),
    ("Original Data", "Prediction with ture data", "true"),
]


def collect_squared_errors(
    predictor: Predictor2014,
    month_paths: dict[str, str],
) -> tuple[np.ndarray, dict[str, float], list[dict[str, float | str | int]]]:
    squared_errors_by_month = []
    season_mse = {}
    season_rows = []
    for month_name in MONTH_ORDER:
        pred, true, _, _ = predictor.predict_single_month(Path(month_paths[month_name]))
        squared_errors = (pred - true) ** 2
        mse = float(np.mean(squared_errors))
        variance = float(np.var(squared_errors, ddof=0))
        season = SEASON_BY_MONTH[month_name]
        squared_errors_by_month.append(squared_errors)
        season_mse[f"{season.lower()}_mse"] = mse
        season_rows.append(
            {
                "season": season,
                "month": month_name,
                "n_prediction_points": int(squared_errors.size),
                "mse": mse,
                "squared_error_variance": variance,
                "squared_error_std": float(np.sqrt(variance)),
            }
        )
    return np.concatenate(squared_errors_by_month), season_mse, season_rows


def main() -> None:
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    sample_manifest = summary["sample_manifest"]
    predictor = Predictor2014()
    rows = []
    season_rows = []

    for rate in RATES:
        for method_key, method_label, manifest_key in METHOD_CONFIGS:
            if manifest_key == "true":
                month_paths = sample_manifest["true"]
            else:
                month_paths = sample_manifest[manifest_key][rate]

            squared_errors, season_mse, method_season_rows = collect_squared_errors(
                predictor, month_paths
            )
            mse = float(np.mean(squared_errors))
            squared_error_variance = float(np.var(squared_errors, ddof=0))

            for season_row in method_season_rows:
                season_rows.append(
                    {
                        "missing_rate": rate,
                        "method_key": method_key,
                        "method": method_label,
                        **season_row,
                    }
                )

            rows.append(
                {
                    "missing_rate": rate,
                    "method_key": method_key,
                    "method": method_label,
                    "n_prediction_points": int(squared_errors.size),
                    "mse_all_points": mse,
                    "squared_error_variance_all_points": squared_error_variance,
                    "squared_error_std_all_points": float(np.sqrt(squared_error_variance)),
                    **season_mse,
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    season_output_path = SEASON_OUTPUT_PATH
    try:
        season_file = SEASON_OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        season_output_path = SEASON_FALLBACK_OUTPUT_PATH
        season_file = SEASON_FALLBACK_OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig")

    with season_file as f:
        writer = csv.DictWriter(f, fieldnames=list(season_rows[0].keys()))
        writer.writeheader()
        writer.writerows(season_rows)

    print(f"Saved all-point four-season MSE and variance to: {OUTPUT_PATH}")
    print(f"Saved per-season MSE and variance to: {season_output_path}")


if __name__ == "__main__":
    main()
