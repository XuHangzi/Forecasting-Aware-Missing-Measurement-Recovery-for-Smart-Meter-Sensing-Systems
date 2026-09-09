# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import importlib.util
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
GAIN_DIR = PROJECT_ROOT / "GAIN_pytorch"
LSTM_DIR = PROJECT_ROOT / "LSTM"
REVIEW_DIR = PROJECT_ROOT / "review_training_hyperparam_experiments"

USER_COLUMNS = ["user_3", "user_4", "user_5", "user_6"]
USER_COLUMN = "user_3"
USER_INDEX = USER_COLUMNS.index(USER_COLUMN)

LAMBDA_2_VALUES = [1.0, 10.0, 50.0, 100.0, 200.0, 500.0]
LAMBDA_3_VALUES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

MISS_RATE = 0.2
SEED = 42
STRIDE_HOURS = 24
PREDICTION_BATCH_SIZE = 128

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


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_lambda(value: float) -> str:
    if 0 < abs(value) < 0.1:
        return f"{value:.0e}"
    return f"{value:g}"


def load_review_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    lambda2_path = REVIEW_DIR / "results" / "lambda_2_sensitivity.csv"
    lambda3_path = REVIEW_DIR / "results" / "lambda_3_sensitivity.csv"
    if not lambda2_path.exists() or not lambda3_path.exists():
        raise FileNotFoundError("Missing review hyperparameter result CSV files.")
    return pd.read_csv(lambda2_path), pd.read_csv(lambda3_path)


def row_for_value(df: pd.DataFrame, key: str, value: float) -> dict[str, Any]:
    matched = df[np.isclose(df[key].astype(float), value)]
    if matched.empty:
        raise ValueError(f"Missing row for {key}={value}")
    return matched.iloc[0].to_dict()


def lambda3_model_dir(value: float) -> Path:
    return REVIEW_DIR / "models" / f"lambda_{value:g}".replace(".", "p")


def row_for_lambda3_value(df: pd.DataFrame, value: float) -> dict[str, Any]:
    matched = df[np.isclose(df["lambda_3"].astype(float), value)]
    if not matched.empty:
        return matched.iloc[0].to_dict()

    model_dir = lambda3_model_dir(value)
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        raise ValueError(
            f"Missing row and trained review model for lambda_3={value}: {metrics_path}"
        )
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "lambda_2": float(metrics.get("alpha", 100.0)),
        "lambda_3": float(metrics.get("prediction_weight", value)),
        "test_prediction_mse": float(metrics["test_prediction_mse"]),
        "imputation_rmse": float(metrics["imputation_rmse"]),
        "best_iteration": metrics.get("best_iteration"),
        "model_dir": str(model_dir),
    }


def load_2013_truth() -> pd.DataFrame:
    source_path = GAIN_DIR / "data" / "electricity_data_2013.csv"
    df = pd.read_csv(source_path)[["hour"] + USER_COLUMNS].copy()
    return df


def reconstruct_review_data_m(raw_values: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return (rng.uniform(size=raw_values.shape) < (1 - MISS_RATE)).astype(np.float32)


def normalized_user_missing_rmse(
    true_values: np.ndarray,
    imputed_values: np.ndarray,
    data_m: np.ndarray,
    user_idx: int,
) -> float:
    missing_mask = data_m[:, user_idx] == 0
    true_user = true_values[:, user_idx].astype(np.float64)
    imputed_user = imputed_values[:, user_idx].astype(np.float64)
    min_val = np.nanmin(true_user)
    max_range = np.nanmax(true_user - min_val)
    true_norm = (true_user - min_val) / (max_range + 1e-6)
    imputed_norm = (imputed_user - min_val) / (max_range + 1e-6)
    return float(np.sqrt(np.mean((true_norm[missing_mask] - imputed_norm[missing_mask]) ** 2)))


class User3FullWindowEvaluator:
    def __init__(self, truth_df: pd.DataFrame) -> None:
        self.module = load_module(LSTM_DIR / "LSTNet_revised.py", "lstnet_review_user3_2013")
        with open(LSTM_DIR / "lstnet_scalers_optimized.pkl", "rb") as f:
            self.scalers = pickle.load(f)
        self.device = self.module.config.device
        self.model = self.module.LSTNetOptimized(n_users=len(self.scalers))
        self.model.load_state_dict(
            torch.load(
                LSTM_DIR / "lstnet_best_model_optimized.pth",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model.to(self.device)
        self.model.eval()

        self.lookback = int(self.module.config.lookback)
        self.horizon = int(self.module.config.horizon)
        self.timestamps = truth_df["hour"].astype(str).to_numpy()
        self.time_features = self.module.create_time_features(self.timestamps).astype(np.float32)
        max_start = len(truth_df) - self.lookback - self.horizon
        if max_start < 0:
            raise ValueError("2013 data is too short for the LSTM lookback and horizon.")
        self.starts = np.arange(0, max_start + 1, STRIDE_HOURS, dtype=np.int64)

    def build_user3_windows(
        self,
        imputed_df: pd.DataFrame,
        truth_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        feature_dim = 14
        x = np.empty((len(self.starts), self.lookback, feature_dim), dtype=np.float32)
        y_true = np.empty((len(self.starts), self.horizon), dtype=np.float32)

        imputed_series = imputed_df[USER_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)
        true_series = truth_df[USER_COLUMN].to_numpy(dtype=np.float32)
        normalized_series = self.scalers[USER_INDEX].transform(imputed_series).flatten()
        stat_features = self.module.add_stat_features(
            normalized_series,
            self.lookback,
        ).astype(np.float32)

        for row_idx, start in enumerate(self.starts):
            end = start + self.lookback
            target_end = end + self.horizon
            x_elec = normalized_series[start:end].reshape(-1, 1).astype(np.float32)
            x[row_idx] = np.concatenate(
                [
                    x_elec,
                    self.time_features[start:end],
                    stat_features[start:end],
                ],
                axis=1,
            )
            y_true[row_idx] = true_series[end:target_end]
        return x, y_true

    def prediction_mse(self, imputed_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict[str, Any]:
        x, y_true = self.build_user3_windows(imputed_df, truth_df)
        total_sq_error = 0.0
        total_count = 0

        with torch.no_grad():
            for start in range(0, len(x), PREDICTION_BATCH_SIZE):
                end = min(start + PREDICTION_BATCH_SIZE, len(x))
                x_batch = torch.from_numpy(x[start:end]).to(self.device)
                users_batch = torch.full(
                    (end - start,),
                    USER_INDEX,
                    dtype=torch.long,
                    device=self.device,
                )
                pred_norm = self.model(x_batch, users_batch).squeeze(-1).cpu().numpy()
                pred_raw = self.scalers[USER_INDEX].inverse_transform(
                    pred_norm.reshape(-1, 1)
                ).reshape(end - start, self.horizon)
                sq_error = (pred_raw.astype(np.float64) - y_true[start:end]) ** 2
                total_sq_error += float(np.sum(sq_error))
                total_count += int(sq_error.size)

        return {
            "user3_full_window_prediction_mse_2013": total_sq_error / total_count,
            "prediction_windows": int(len(x)),
            "target_points": int(total_count),
            "windows_per_user": int(len(self.starts)),
            "stride_hours": STRIDE_HOURS,
        }


def load_review_imputed(row: dict[str, Any], truth_df: pd.DataFrame) -> pd.DataFrame:
    model_dir = Path(str(row["model_dir"]))
    imputed_path = model_dir / "guided_imputed_processed.csv"
    if not imputed_path.exists():
        raise FileNotFoundError(f"Missing review imputed data: {imputed_path}")
    imputed_df = pd.read_csv(imputed_path)
    missing_cols = [col for col in USER_COLUMNS if col not in imputed_df.columns]
    if missing_cols:
        raise ValueError(f"Missing user columns in {imputed_path}: {missing_cols}")
    if len(imputed_df) != len(truth_df):
        raise ValueError(f"Row count mismatch for {imputed_path}")
    imputed_df = imputed_df[USER_COLUMNS].copy()
    imputed_df.insert(0, "hour", truth_df["hour"].to_numpy())
    return imputed_df


def evaluate_review_model_row(
    row: dict[str, Any],
    evaluator: User3FullWindowEvaluator,
    truth_df: pd.DataFrame,
    true_values: np.ndarray,
    data_m: np.ndarray,
) -> dict[str, Any]:
    lambda2 = float(row["lambda_2"])
    lambda3 = float(row["lambda_3"])
    print(f"Evaluating 2013 review output lambda_2={lambda2:g}, lambda_3={lambda3:g}")
    imputed_df = load_review_imputed(row, truth_df)
    imputed_values = imputed_df[USER_COLUMNS].to_numpy(dtype=np.float32)
    prediction_metrics = evaluator.prediction_mse(imputed_df, truth_df)

    return {
        "lambda_2": lambda2,
        "lambda_3": lambda3,
        "user3_full_window_prediction_mse_2013": prediction_metrics[
            "user3_full_window_prediction_mse_2013"
        ],
        "user3_normalized_missing_imputation_rmse_2013": normalized_user_missing_rmse(
            true_values,
            imputed_values,
            data_m,
            USER_INDEX,
        ),
        "prediction_windows": prediction_metrics["prediction_windows"],
        "target_points": prediction_metrics["target_points"],
        "windows_per_user": prediction_metrics["windows_per_user"],
        "stride_hours": prediction_metrics["stride_hours"],
        "review_test_prediction_mse_2013": float(row["test_prediction_mse"]),
        "review_imputation_rmse_2013": float(row["imputation_rmse"]),
        "review_best_iteration": row.get("best_iteration"),
        "review_model_dir": str(row["model_dir"]),
        "review_imputed_processed": str(Path(str(row["model_dir"])) / "guided_imputed_processed.csv"),
    }


def plot_sensitivity(rows: list[dict[str, Any]], x_key: str, output_path: Path) -> None:
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
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    truth_df = load_2013_truth()
    true_values = truth_df[USER_COLUMNS].to_numpy(dtype=np.float32)
    data_m = reconstruct_review_data_m(true_values)
    evaluator = User3FullWindowEvaluator(truth_df)
    lambda2_source, lambda3_source = load_review_rows()

    lambda2_rows = [
        evaluate_review_model_row(
            row_for_value(lambda2_source, "lambda_2", value),
            evaluator,
            truth_df,
            true_values,
            data_m,
        )
        for value in LAMBDA_2_VALUES
    ]
    lambda3_rows = [
        evaluate_review_model_row(
            row_for_lambda3_value(lambda3_source, value),
            evaluator,
            truth_df,
            true_values,
            data_m,
        )
        for value in LAMBDA_3_VALUES
    ]

    lambda2_csv = RESULTS_DIR / "lambda_2_user3_fullwindow_2013.csv"
    lambda3_csv = RESULTS_DIR / "lambda_3_user3_fullwindow_2013.csv"
    write_csv(lambda2_csv, lambda2_rows)
    write_csv(lambda3_csv, lambda3_rows)

    lambda2_fig = FIGURES_DIR / "lambda_2_user3_fullwindow_2013.jpeg"
    lambda3_fig = FIGURES_DIR / "lambda_3_user3_fullwindow_2013.jpeg"
    plot_sensitivity(lambda2_rows, "lambda_2", lambda2_fig)
    plot_sensitivity(lambda3_rows, "lambda_3", lambda3_fig)

    review_fig_dir = REVIEW_DIR / "figures"
    review_fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lambda2_fig, review_fig_dir / lambda2_fig.name)
    shutil.copy2(lambda3_fig, review_fig_dir / lambda3_fig.name)

    with (RESULTS_DIR / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "settings": {
                    "method": "saved review guided_imputed_processed.csv + direct user3 2013 full-window LSTM inference",
                    "truth_data": str(GAIN_DIR / "data" / "electricity_data_2013.csv"),
                    "review_model_source": str(REVIEW_DIR / "models"),
                    "user": USER_COLUMN,
                    "miss_rate": MISS_RATE,
                    "seed": SEED,
                    "lookback": evaluator.lookback,
                    "horizon": evaluator.horizon,
                    "stride_hours": STRIDE_HOURS,
                    "windows": int(len(evaluator.starts)),
                    "target_points": int(len(evaluator.starts) * evaluator.horizon),
                    "device": str(evaluator.device),
                },
                "lambda_2_sensitivity": lambda2_rows,
                "lambda_3_sensitivity": lambda3_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Completed review-model user3 full-window 2013 experiment.")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Copied figures to: {review_fig_dir}")


if __name__ == "__main__":
    main()
