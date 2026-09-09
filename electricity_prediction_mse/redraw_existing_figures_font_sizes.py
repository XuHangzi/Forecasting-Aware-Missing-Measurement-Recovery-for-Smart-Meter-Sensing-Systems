# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = EXP_DIR / "predictions"
PLOTS_DIR = EXP_DIR / "plots"

SEASON_MAP = {
    "January": "Winter",
    "April": "Spring",
    "July": "Summer",
    "November": "Autumn",
}

PRED_AXIS_LABEL_FONTSIZE = 20
BAR_AXIS_LABEL_FONTSIZE = 22
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 16


def redraw_prediction_comparison() -> None:
    prediction_json = PREDICTIONS_DIR / "prediction_results_rate_0.2.json"
    with prediction_json.open("r", encoding="utf-8") as f:
        results = json.load(f)

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "KaiTi", "STKaiti"],
            "axes.unicode_minus": False,
            "axes.labelpad": 10,
            "axes.titlepad": 15,
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )

    line_styles = {
        "True Value": {
            "color": "#2E86AB",
            "linewidth": 2.5,
            "alpha": 0.9,
            "label": "True value",
        },
        "True Data": {
            "color": "#E63946",
            "linewidth": 2.0,
            "alpha": 0.8,
            "marker": "o",
            "markevery": 8,
            "label": "Prediction with ture data",
        },
        "Missing Data": {
            "color": "#F77F00",
            "linewidth": 2.0,
            "alpha": 0.8,
            "marker": "s",
            "markevery": 8,
            "label": "Prediction w/o GAIN",
        },
        "GAIN Imputed Data": {
            "color": "#06D6A0",
            "linewidth": 2.0,
            "alpha": 0.8,
            "marker": "^",
            "markevery": 8,
            "label": "Prediction with GAIN",
        },
        "Guided GAIN Data": {
            "color": "#8338EC",
            "linewidth": 2.0,
            "alpha": 0.8,
            "marker": "D",
            "markevery": 8,
            "label": "Prediction with fine-tuned GAIN",
        },
    }

    plt.figure(figsize=(12, 18))
    month_order = ["January", "April", "July", "November"]
    for i, month_name in enumerate(month_order):
        season = SEASON_MAP[month_name]
        plt.subplot(4, 1, i + 1)

        true_values = np.array(results["True Data"][month_name]["true"])
        hours = np.arange(len(true_values))
        plt.plot(hours, true_values, **line_styles["True Value"])

        for data_type in ["True Data", "Missing Data", "GAIN Imputed Data", "Guided GAIN Data"]:
            pred_values = np.array(results[data_type][month_name]["pred"])
            plt.plot(hours, pred_values, **line_styles[data_type])

        all_series = [true_values] + [
            np.array(results[data_type][month_name]["pred"])
            for data_type in ["True Data", "Missing Data", "GAIN Imputed Data", "Guided GAIN Data"]
        ]
        y_max = max(series.max() for series in all_series)
        y_min = min(series.min() for series in all_series)
        y_range = max(y_max - y_min, 1e-6)
        plt.ylim(y_min - 0.08 * y_range, y_max + 1.50 * y_range)

        plt.xlabel(
            f"{season}: Time steps in the predicted 7 days",
            fontsize=PRED_AXIS_LABEL_FONTSIZE,
        )
        plt.ylabel(
            "Electricity Consumption (kWh)",
            fontsize=PRED_AXIS_LABEL_FONTSIZE,
            labelpad=26,
        )
        plt.legend(
            fontsize=LEGEND_FONTSIZE,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            borderaxespad=0.2,
            ncol=1,
            frameon=True,
        )
        plt.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        plt.grid(alpha=0.3, linestyle="--")
        plt.xlim(0, len(true_values) - 1)

    plt.tight_layout(pad=3.4, h_pad=5.5)
    plt.savefig(
        PREDICTIONS_DIR / "LSTnet_prediction_comparison_with_guided_0.2.jpeg",
        dpi=300,
        bbox_inches="tight",
        format="jpeg",
    )
    plt.close()


def redraw_mse_bars() -> None:
    with (EXP_DIR / "experiment_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    mse_results = summary["mse_results"]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    colors = {
        "Missing Data": "#FF9999",
        "GAIN Imputed Data": "#66B2FF",
        "Guided GAIN Data": "#FFD166",
        "Original Data": "#99FF99",
    }

    rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    for season, data in mse_results.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        df = pd.DataFrame(
            {
                "Missing Rate": rates,
                "Missing Data": data["Missing Data"],
                "GAIN Imputed Data": data["GAIN Imputed Data"],
                "Guided GAIN Data": data["Guided GAIN Data"],
                "Original Data": data["Original Data"],
            }
        )

        x = np.arange(len(df["Missing Rate"]))
        width = 0.2

        ax.bar(
            x - 1.5 * width,
            df["Missing Data"],
            width,
            label="Prediction w/o GAIN",
            color=colors["Missing Data"],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            x - 0.5 * width,
            df["GAIN Imputed Data"],
            width,
            label="Prediction with GAIN",
            color=colors["GAIN Imputed Data"],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            x + 0.5 * width,
            df["Guided GAIN Data"],
            width,
            label="Prediction with fine-tuned GAIN",
            color=colors["Guided GAIN Data"],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            x + 1.5 * width,
            df["Original Data"],
            width,
            label="Prediction with ture data",
            color=colors["Original Data"],
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_xlabel("Missing Rate", fontsize=BAR_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Mean Squared Error", fontsize=BAR_AXIS_LABEL_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{rate:.1f}" for rate in df["Missing Rate"]], fontsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        max_val = max(
            df["Missing Data"].max(),
            df["GAIN Imputed Data"].max(),
            df["Guided GAIN Data"].max(),
            df["Original Data"].max(),
        )
        if season in {"Winter", "Summer", "Autumn"}:
            ax.set_ylim(0, max_val * 1.1)
        else:
            ax.set_ylim(0, max(200, max_val * 1.1))

        ax.legend(fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        fig.savefig(
            PLOTS_DIR / f"electricity_prediction_mse_{season}_missing_rates_with_guided.jpeg",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="jpeg",
        )
        plt.close(fig)


def main() -> None:
    redraw_prediction_comparison()
    redraw_mse_bars()
    print("Redrew existing 2014 figures with updated font sizes only.")


if __name__ == "__main__":
    main()
