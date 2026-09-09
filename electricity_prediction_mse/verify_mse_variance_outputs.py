# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
RATES = ["0.1", "0.2", "0.3", "0.4", "0.5"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
METHOD_MAP = {
    "Missing Data": "Prediction w/o GAIN",
    "GAIN Imputed Data": "Prediction with GAIN",
    "Guided GAIN Data": "Prediction with fine-tuned GAIN",
    "Original Data": "Prediction with ture data",
}


def main() -> None:
    season_path = EXP_DIR / "plots" / "season_mse_variance_full.csv"
    if not season_path.exists():
        season_path = EXP_DIR / "plots" / "season_mse_variance.csv"
    season = pd.read_csv(season_path)
    all_points = pd.read_csv(EXP_DIR / "plots" / "four_season_all_points_mse_variance.csv")
    with (EXP_DIR / "experiment_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    diffs = []
    all_points_season_diffs = []
    all_points_mean_diffs = []
    for season_name in SEASONS:
        for method_key, method_label in METHOD_MAP.items():
            for rate_index, rate in enumerate(RATES):
                csv_value = float(
                    season[
                        (season["missing_rate"] == float(rate))
                        & (season["season"] == season_name)
                        & (season["method"] == method_label)
                    ]["mse"].iloc[0]
                )
                summary_value = float(summary["mse_results"][season_name][method_key][rate_index])
                diffs.append(abs(csv_value - summary_value))

                all_points_row = all_points[
                    (all_points["missing_rate"] == float(rate))
                    & (all_points["method"] == method_label)
                ].iloc[0]
                all_points_value = float(all_points_row[f"{season_name.lower()}_mse"])
                all_points_season_diffs.append(abs(all_points_value - summary_value))

    for method_key, method_label in METHOD_MAP.items():
        for rate_index, rate in enumerate(RATES):
            source_values = [
                float(summary["mse_results"][season_name][method_key][rate_index])
                for season_name in SEASONS
            ]
            all_points_row = all_points[
                (all_points["missing_rate"] == float(rate))
                & (all_points["method"] == method_label)
            ].iloc[0]
            all_points_mean_diffs.append(
                abs(float(all_points_row["mse_all_points"]) - float(np.mean(source_values)))
            )

    with (EXP_DIR / "predictions" / "prediction_results_rate_0.2.json").open(
        "r", encoding="utf-8"
    ) as f:
        prediction_results = json.load(f)

    winter_errors = np.array(prediction_results["Guided GAIN Data"]["January"]["pred"]) - np.array(
        prediction_results["Guided GAIN Data"]["January"]["true"]
    )
    winter_squared_errors = winter_errors**2
    winter_row = season[
        (season["missing_rate"] == 0.2)
        & (season["season"] == "Winter")
        & (season["method"] == "Prediction with fine-tuned GAIN")
    ].iloc[0]

    all_squared_errors = []
    for month_name in ["January", "April", "July", "November"]:
        month_errors = np.array(prediction_results["Guided GAIN Data"][month_name]["pred"]) - np.array(
            prediction_results["Guided GAIN Data"][month_name]["true"]
        )
        all_squared_errors.extend((month_errors**2).tolist())
    all_squared_errors = np.array(all_squared_errors)
    all_row = all_points[
        (all_points["missing_rate"] == 0.2)
        & (all_points["method"] == "Prediction with fine-tuned GAIN")
    ].iloc[0]

    print(f"max_abs_diff_csv_mse_vs_summary={max(diffs):.12g}")
    print(
        "max_abs_diff_all_points_season_columns_vs_summary="
        f"{max(all_points_season_diffs):.12g}"
    )
    print(
        "max_abs_diff_all_points_mse_vs_mean_of_four_seasons="
        f"{max(all_points_mean_diffs):.12g}"
    )
    print(f"season_rows={len(season)} unique_n={sorted(season['n_prediction_points'].unique().tolist())}")
    print(
        f"all_points_rows={len(all_points)} "
        f"unique_n={sorted(all_points['n_prediction_points'].unique().tolist())}"
    )
    print(
        "rate_0.2_winter_finetuned_from_json: "
        f"mse={np.mean(winter_squared_errors):.12g}, csv_mse={float(winter_row['mse']):.12g}, "
        f"var={np.var(winter_squared_errors, ddof=0):.12g}, "
        f"csv_var={float(winter_row['squared_error_variance']):.12g}"
    )
    print(
        "rate_0.2_allseason_finetuned_from_json: "
        f"mse={np.mean(all_squared_errors):.12g}, "
        f"csv_mse={float(all_row['mse_all_points']):.12g}, "
        f"var={np.var(all_squared_errors, ddof=0):.12g}, "
        f"csv_var={float(all_row['squared_error_variance_all_points']):.12g}"
    )


if __name__ == "__main__":
    main()
