# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"


def configure_plot_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "KaiTi", "STKaiti"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def format_lambda(value: float) -> str:
    if 0 < abs(value) < 0.1:
        return f"{value:.0e}"
    return f"{value:g}"


def plot_sensitivity(rows: list[dict], x_key: str, output_path: Path) -> None:
    configure_plot_fonts()
    labels = [format_lambda(float(row[x_key])) for row in rows]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(7.0, 5.2))
    ax2 = ax1.twinx()
    line1 = ax1.plot(
        x,
        [row["user3_full_window_prediction_mse_2013"] for row in rows],
        marker="o",
        color="#4f81bd",
        label="Prediction MSE",
    )
    line2 = ax2.plot(
        x,
        [row["user3_normalized_missing_imputation_rmse_2013"] for row in rows],
        marker="s",
        color="#c0504d",
        label="Imputation RMSE",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_xlabel(r"$\lambda_2$" if x_key == "lambda_2" else r"$\lambda_3$", fontsize=14)
    ax1.set_ylabel("Prediction MSE", fontsize=14)
    ax2.set_ylabel("Imputation RMSE", fontsize=14)
    ax1.tick_params(axis="y", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)

    pred_values = [row["user3_full_window_prediction_mse_2013"] for row in rows]
    rmse_values = [row["user3_normalized_missing_imputation_rmse_2013"] for row in rows]
    pred_range = max(pred_values) - min(pred_values)
    rmse_range = max(rmse_values) - min(rmse_values)
    ax1.set_ylim(
        min(pred_values) - max(pred_range * 0.08, 1e-6),
        max(pred_values) + max(pred_range * 0.35, 1e-6),
    )
    ax2.set_ylim(
        min(rmse_values) - max(rmse_range * 0.08, 1e-6),
        max(rmse_values) + max(rmse_range * 0.35, 1e-6),
    )
    ax1.grid(alpha=0.25)
    lines = line1 + line2
    ax1.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        frameon=True,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, format="jpeg")
    plt.close(fig)


def main() -> None:
    lambda2_rows = pd.read_csv(RESULTS_DIR / "lambda_2_user3_fullwindow_2013.csv").to_dict(
        orient="records"
    )
    lambda3_rows = pd.read_csv(RESULTS_DIR / "lambda_3_user3_fullwindow_2013.csv").to_dict(
        orient="records"
    )

    plot_sensitivity(
        lambda2_rows,
        "lambda_2",
        FIGURES_DIR / "lambda_2_user3_fullwindow_2013.jpeg",
    )
    plot_sensitivity(
        lambda3_rows,
        "lambda_3",
        FIGURES_DIR / "lambda_3_user3_fullwindow_2013.jpeg",
    )


if __name__ == "__main__":
    main()
