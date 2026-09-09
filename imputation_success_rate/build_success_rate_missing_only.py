# coding=utf-8

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "experiment_2014_4users" / "datasets"
OUTPUT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

USER_COLUMNS = ["user_3", "user_4", "user_5", "user_6"]
MISSING_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_SUCCESS_THRESHOLD = 2.0


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_missing_mask(truth_df: pd.DataFrame, input_df: pd.DataFrame) -> np.ndarray:
    truth_values = truth_df[USER_COLUMNS].to_numpy(dtype=np.float32)
    input_values = input_df[USER_COLUMNS].to_numpy(dtype=np.float32)
    return (input_values == 0) & (truth_values != 0)


def compute_missing_only_accuracy(
    truth_df: pd.DataFrame,
    input_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    threshold: float,
) -> float:
    truth_values = truth_df[USER_COLUMNS].to_numpy(dtype=np.float32)
    imputed_values = imputed_df[USER_COLUMNS].to_numpy(dtype=np.float32)
    missing_mask = compute_missing_mask(truth_df, input_df)

    total_missing = int(missing_mask.sum())
    if total_missing == 0:
        return 0.0

    success_mask = np.abs(imputed_values - truth_values) < threshold
    success_count = int(np.logical_and(success_mask, missing_mask).sum())
    return 100.0 * success_count / total_missing


def main(args):
    truth_path = DATA_ROOT / "true" / "electricity_data_2014_true_4users.csv"
    truth_df = load_table(truth_path)

    gain_rates = []
    guided_rates = []
    rate_labels = []
    detailed = []

    for rate in MISSING_RATES:
        rate_str = f"{rate:.1f}"
        input_path = DATA_ROOT / "missing" / f"electricity_data_2014_missing_{rate_str}_4users.csv"
        gain_path = DATA_ROOT / "gain_imputed" / f"electricity_data_2014_gain_imputed_{rate_str}_4users.csv"
        guided_path = (
            DATA_ROOT / "guided_gain_imputed" / f"electricity_data_2014_guided_gain_imputed_{rate_str}_4users.csv"
        )

        input_df = load_table(input_path)
        gain_df = load_table(gain_path)
        guided_df = load_table(guided_path)

        gain_acc = compute_missing_only_accuracy(truth_df, input_df, gain_df, args.threshold)
        guided_acc = compute_missing_only_accuracy(truth_df, input_df, guided_df, args.threshold)

        gain_rates.append(gain_acc)
        guided_rates.append(guided_acc)
        rate_labels.append(rate)
        detailed.append(
            {
                "missing_rate": rate,
                "success_threshold": args.threshold,
                "gain_accuracy_missing_only": gain_acc,
                "guided_gain_accuracy_missing_only": guided_acc,
            }
        )

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(rate_labels))

    ax.plot(
        x,
        gain_rates,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#1f77b4",
        label="GAIN",
    )
    ax.plot(
        x,
        guided_rates,
        marker="s",
        linewidth=2.2,
        markersize=7,
        color="#d62728",
        label="Fine-tuned GAIN",
    )

    ax.set_xlabel("Missing Rate", fontsize=16)
    ax.set_ylabel("ACC (%)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{rate:.1f}" for rate in rate_labels], fontsize=14)
    all_values = gain_rates + guided_rates
    y_min = max(0.0, np.floor(min(all_values) - 5.0))
    y_max = min(100.0, np.ceil(max(all_values) + 2.0))
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=13, loc="best")
    plt.tight_layout()

    plot_path = OUTPUT_ROOT / f"imputation_success_rate_missing_only{args.output_suffix}_2014_4users.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    summary = {
        "definition": {
            "imputation_accuracy_missing_only": (
                "ACC_missing = (successfully imputed missing entries / total artificially missing entries) * 100."
            ),
            "success_threshold": args.threshold,
            "success_rule": f"A missing position is counted as successful when |imputed - truth| < {args.threshold}.",
        },
        "missing_rates": detailed,
        "plot_path": str(plot_path),
    }

    summary_path = OUTPUT_ROOT / f"imputation_success_missing_only_summary{args.output_suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(detailed, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--output_suffix", type=str, default="")
    main(parser.parse_args())
