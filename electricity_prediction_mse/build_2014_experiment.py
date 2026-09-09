from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAIN_DIR = PROJECT_ROOT / "GAIN_pytorch"
GUIDED_DIR = PROJECT_ROOT / "GAIN_guided"
LSTM_DIR = PROJECT_ROOT / "LSTM"
REFERENCE_DIR = PROJECT_ROOT / "参考"

if str(GAIN_DIR) not in sys.path:
    sys.path.insert(0, str(GAIN_DIR))

from gain import Generator
from utils import normalization, renormalization


RATE_TO_SOURCE = {
    0.1: PROJECT_ROOT / "guided_gain_imputed" / "electricity_data_2014_processed_missing_0.1_complete.csv",
    0.2: PROJECT_ROOT / "guided_gain_imputed" / "electricity_data_2014_processed_missing_complete.csv",
    0.3: PROJECT_ROOT / "guided_gain_imputed" / "electricity_data_2014_processed_missing_0.3_complete.csv",
    0.4: PROJECT_ROOT / "guided_gain_imputed" / "electricity_data_2014_processed_missing_0.4_complete.csv",
    0.5: PROJECT_ROOT / "guided_gain_imputed" / "electricity_data_2014_processed_missing_0.5_complete.csv",
}

USER_COLUMN = "user_3"
USER_LABEL = "user3"
USER_COLUMNS = ["user_3", "user_4", "user_5", "user_6"]
USER_INDEX = USER_COLUMNS.index(USER_COLUMN)
MONTHS = {
    "January": (1, "jan"),
    "April": (4, "apr"),
    "July": (7, "jul"),
    "November": (11, "nov"),
}
SEASON_MAP = {
    "January": "Winter",
    "April": "Spring",
    "July": "Summer",
    "November": "Autumn",
}

LOOKBACK = 24 * 30
HORIZON = 24 * 7


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_generator(weight_path: Path, n_users: int, device: torch.device) -> Generator:
    generator = Generator(n_users, n_users).to(device)
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def infer_missing_mask(reference_values: np.ndarray, incomplete_values: np.ndarray) -> np.ndarray:
    return np.isnan(incomplete_values) | ((incomplete_values == 0) & (reference_values != 0))


def impute_with_generator(
    generator: Generator,
    incomplete_values: np.ndarray,
    missing_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    miss_data = incomplete_values.astype(np.float32).copy()
    miss_data[missing_mask] = np.nan

    norm_data, norm_parameters = normalization(miss_data)
    observed_norm = np.nan_to_num(norm_data, nan=0.0).astype(np.float32)
    data_m = (~np.isnan(miss_data)).astype(np.float32)

    observed_tensor = torch.tensor(observed_norm, dtype=torch.float32, device=device)
    data_m_tensor = torch.tensor(data_m, dtype=torch.float32, device=device)
    noise_tensor = torch.rand_like(observed_tensor) * 0.01

    full_x = data_m_tensor * observed_tensor + (1 - data_m_tensor) * noise_tensor
    with torch.no_grad():
        generated = generator(full_x, data_m_tensor).cpu().numpy()

    imputed_norm = data_m * observed_norm + (1 - data_m) * generated
    imputed_raw = renormalization(imputed_norm, norm_parameters)
    return imputed_raw.astype(np.float32)


def prepare_datasets(output_root: Path, device: torch.device) -> dict:
    datasets_dir = output_root / "datasets"
    true_dir = datasets_dir / "true"
    missing_dir = datasets_dir / "missing"
    gain_dir = datasets_dir / "gain_imputed"
    guided_dir = datasets_dir / "guided_gain_imputed"
    for path in [true_dir, missing_dir, gain_dir, guided_dir]:
        path.mkdir(parents=True, exist_ok=True)

    truth_path = GAIN_DIR / "data" / "electricity_data_2014_原版.csv"
    truth_df = pd.read_csv(truth_path)[["hour"] + USER_COLUMNS].copy()
    truth_output = true_dir / "electricity_data_2014_true_4users.csv"
    truth_df.to_csv(truth_output, index=False)

    reference_values = truth_df[USER_COLUMNS].to_numpy(dtype=np.float32)
    baseline_generator = load_generator(GAIN_DIR / "saved_models" / "generator.pth", len(USER_COLUMNS), device)
    guided_generator = load_generator(
        GUIDED_DIR / "saved_models_guided" / "guided_generator_best.pth",
        len(USER_COLUMNS),
        device,
    )

    dataset_manifest = {
        "truth": str(truth_output),
        "rates": {},
    }

    for rate, source_path in RATE_TO_SOURCE.items():
        if source_path.exists():
            source_df = pd.read_csv(source_path)[["hour"] + USER_COLUMNS].copy()
            source_origin = str(source_path)
        else:
            rng = np.random.default_rng(20240511 + int(rate * 1000))
            source_df = truth_df.copy()
            synthetic_values = source_df[USER_COLUMNS].to_numpy(dtype=np.float32).copy()
            missing_mask = rng.uniform(size=synthetic_values.shape) < rate
            synthetic_values[missing_mask] = 0.0
            source_df.loc[:, USER_COLUMNS] = synthetic_values
            source_origin = "synthetic_from_truth"
        missing_output = missing_dir / f"electricity_data_2014_missing_{rate:.1f}_4users.csv"
        source_df.to_csv(missing_output, index=False)

        incomplete_values = source_df[USER_COLUMNS].to_numpy(dtype=np.float32)
        missing_mask = infer_missing_mask(reference_values, incomplete_values)

        gain_imputed_values = impute_with_generator(
            baseline_generator, incomplete_values, missing_mask, device
        )
        gain_imputed_df = pd.DataFrame(gain_imputed_values, columns=USER_COLUMNS)
        gain_imputed_df.insert(0, "hour", source_df["hour"].to_numpy())
        gain_output = gain_dir / f"electricity_data_2014_gain_imputed_{rate:.1f}_4users.csv"
        gain_imputed_df.to_csv(gain_output, index=False)

        guided_imputed_values = impute_with_generator(
            guided_generator, incomplete_values, missing_mask, device
        )
        guided_imputed_df = pd.DataFrame(guided_imputed_values, columns=USER_COLUMNS)
        guided_imputed_df.insert(0, "hour", source_df["hour"].to_numpy())
        guided_output = guided_dir / f"electricity_data_2014_guided_gain_imputed_{rate:.1f}_4users.csv"
        guided_imputed_df.to_csv(guided_output, index=False)

        dataset_manifest["rates"][f"{rate:.1f}"] = {
            "source_origin": source_origin,
            "missing": str(missing_output),
            "gain_imputed": str(gain_output),
            "guided_gain_imputed": str(guided_output),
            "missing_count": int(missing_mask.sum()),
        }

    return dataset_manifest


def extract_sample_window(df: pd.DataFrame, month: int) -> pd.DataFrame:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    month_data = data[data["timestamp"].dt.month == month].sort_values("timestamp").reset_index(drop=True)
    if len(month_data) < LOOKBACK:
        raise ValueError(f"Month {month} does not contain {LOOKBACK} hourly samples.")

    input_data = month_data.iloc[:LOOKBACK].copy()
    last_input_ts = input_data["timestamp"].iloc[-1]
    true_data = data[data["timestamp"] > last_input_ts].sort_values("timestamp").iloc[:HORIZON].copy()
    if len(true_data) < HORIZON:
        raise ValueError(f"Month {month} does not have {HORIZON} hours of ground-truth horizon.")

    combined = pd.concat([input_data, true_data], ignore_index=True)
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H")
    return combined


def create_sample_files(output_root: Path, dataset_manifest: dict) -> dict:
    samples_dir = output_root / "samples"
    true_dir = samples_dir / "true"
    true_dir.mkdir(parents=True, exist_ok=True)

    sample_manifest = {
        "true": {},
        "missing": {},
        "gain_imputed": {},
        "guided_gain_imputed": {},
    }

    truth_df = pd.read_csv(dataset_manifest["truth"])[["hour", USER_COLUMN]].rename(
        columns={"hour": "timestamp", USER_COLUMN: "consumption"}
    )

    for month_name, (month_num, month_short) in MONTHS.items():
        sample_df = extract_sample_window(truth_df, month_num)
        sample_path = true_dir / f"{USER_LABEL}_{month_short}_30days_input_7days_true.csv"
        sample_df.to_csv(sample_path, index=False)
        sample_manifest["true"][month_name] = str(sample_path)

    for data_type in ["missing", "gain_imputed", "guided_gain_imputed"]:
        for rate, paths in dataset_manifest["rates"].items():
            rate_dir = samples_dir / data_type / f"rate_{rate}"
            rate_dir.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(paths[data_type])[["hour", USER_COLUMN]].rename(
                columns={"hour": "timestamp", USER_COLUMN: "consumption"}
            )
            sample_manifest[data_type].setdefault(rate, {})
            for month_name, (month_num, month_short) in MONTHS.items():
                sample_df = extract_sample_window(df, month_num)
                sample_path = rate_dir / f"{USER_LABEL}_{month_short}_30days_input_7days_{data_type}_{rate}.csv"
                sample_df.to_csv(sample_path, index=False)
                sample_manifest[data_type][rate][month_name] = str(sample_path)

    return sample_manifest


class Predictor2014:
    def __init__(self) -> None:
        self.module = load_module(LSTM_DIR / "LSTNet_revised.py", "lstm_predict2014_module")
        self.model, self.scalers, self.n_users = self.load_model_and_scalers()

    def load_model_and_scalers(self):
        with open(LSTM_DIR / "lstnet_scalers_optimized.pkl", "rb") as f:
            scalers = pickle.load(f)
        model = self.module.LSTNetOptimized(n_users=len(scalers))
        model.load_state_dict(torch.load(LSTM_DIR / "lstnet_best_model_optimized.pth", map_location=self.module.config.device))
        model.to(self.module.config.device)
        model.eval()
        return model, scalers, len(scalers)

    def predict_single_month(self, csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        input_data = df.iloc[: self.module.config.lookback].copy()
        true_data = df.iloc[
            self.module.config.lookback : self.module.config.lookback + self.module.config.horizon
        ].copy()

        timestamps_input = input_data["timestamp"].dt.strftime("%Y-%m-%d %H").to_numpy()
        consumption_input = input_data["consumption"].to_numpy().reshape(-1, 1)
        scaler = self.scalers[USER_INDEX]
        normalized_consumption = scaler.transform(consumption_input).flatten()

        time_features = self.module.create_time_features(timestamps_input)
        stat_features = self.module.add_stat_features(normalized_consumption, self.module.config.lookback)
        x = np.concatenate(
            [
                normalized_consumption.reshape(-1, 1),
                time_features,
                stat_features,
            ],
            axis=1,
        ).reshape(1, self.module.config.lookback, -1)

        x_tensor = torch.FloatTensor(x).to(self.module.config.device)
        user_idx_tensor = torch.tensor([USER_INDEX], dtype=torch.long).to(self.module.config.device)
        with torch.no_grad():
            pred_normalized = self.model(x_tensor, user_idx_tensor).cpu().numpy()

        pred_values = scaler.inverse_transform(pred_normalized[0]).flatten()
        true_values = true_data["consumption"].to_numpy()
        pred_timestamps = true_data["timestamp"].to_numpy()
        mse = float(np.mean((pred_values - true_values) ** 2))
        return pred_values, true_values, pred_timestamps, mse


def plot_prediction_comparison(
    output_root: Path,
    sample_manifest: dict,
    predictor: Predictor2014,
) -> dict:
    predictions_dir = output_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    rate = "0.2"
    data_configs = {
        "True Data": sample_manifest["true"],
        "Missing Data": sample_manifest["missing"][rate],
        "GAIN Imputed Data": sample_manifest["gain_imputed"][rate],
        "Guided GAIN Data": sample_manifest["guided_gain_imputed"][rate],
    }

    results = {}
    for data_type, month_config in data_configs.items():
        results[data_type] = {}
        for month_name, csv_path in month_config.items():
            pred, true, ts, mse = predictor.predict_single_month(Path(csv_path))
            results[data_type][month_name] = {
                "pred": pred.tolist(),
                "true": true.tolist(),
                "mse": mse,
            }

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

        plt.xlabel(f"{season}: Time steps in the predicted 7 days", fontsize=20)
        plt.ylabel(
            "Electricity Consumption (kWh)",
            fontsize=20,
            labelpad=26,
        )
        plt.legend(
            fontsize=16,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            borderaxespad=0.2,
            ncol=1,
            frameon=True,
        )
        plt.tick_params(axis="both", labelsize=20)
        plt.grid(alpha=0.3, linestyle="--")
        plt.xlim(0, len(true_values) - 1)

    plt.tight_layout(pad=3.4, h_pad=5.5)
    plot_path = predictions_dir / "LSTnet_prediction_comparison_with_guided_0.2.jpeg"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", format="jpeg")
    plt.close()

    with open(predictions_dir / "prediction_results_rate_0.2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def compute_all_mse(sample_manifest: dict, predictor: Predictor2014) -> dict:
    mse_results = {
        season: {
            "Missing Data": [],
            "GAIN Imputed Data": [],
            "Guided GAIN Data": [],
            "Original Data": [],
        }
        for season in SEASON_MAP.values()
    }

    month_order = ["January", "April", "July", "November"]
    rates = ["0.1", "0.2", "0.3", "0.4", "0.5"]

    original_mse = {}
    for month_name in month_order:
        _, _, _, mse = predictor.predict_single_month(Path(sample_manifest["true"][month_name]))
        original_mse[month_name] = mse

    for rate in rates:
        for month_name in month_order:
            season = SEASON_MAP[month_name]

            _, _, _, mse_missing = predictor.predict_single_month(
                Path(sample_manifest["missing"][rate][month_name])
            )
            _, _, _, mse_gain = predictor.predict_single_month(
                Path(sample_manifest["gain_imputed"][rate][month_name])
            )
            _, _, _, mse_guided = predictor.predict_single_month(
                Path(sample_manifest["guided_gain_imputed"][rate][month_name])
            )

            mse_results[season]["Missing Data"].append(mse_missing)
            mse_results[season]["GAIN Imputed Data"].append(mse_gain)
            mse_results[season]["Guided GAIN Data"].append(mse_guided)
            mse_results[season]["Original Data"].append(original_mse[month_name])

    return mse_results


def plot_mse_bars(output_root: Path, mse_results: dict) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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

        ax.set_xlabel("Missing Rate", fontsize=22)
        ax.set_ylabel("Mean Squared Error", fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{rate:.1f}" for rate in df["Missing Rate"]], fontsize=20)
        ax.tick_params(axis="y", labelsize=20)
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

        ax.legend(fontsize=16)
        plt.tight_layout()
        fig.savefig(
            plots_dir / f"electricity_prediction_mse_{season}_missing_rates_with_guided.jpeg",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="jpeg",
        )
        plt.close(fig)


def main():
    output_root = PROJECT_ROOT / "experiment_2014_4users"
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_manifest = prepare_datasets(output_root, device)
    sample_manifest = create_sample_files(output_root, dataset_manifest)
    predictor = Predictor2014()
    prediction_results = plot_prediction_comparison(output_root, sample_manifest, predictor)
    mse_results = compute_all_mse(sample_manifest, predictor)
    plot_mse_bars(output_root, mse_results)

    summary = {
        "user_column": USER_COLUMN,
        "user_index_for_lstm": USER_INDEX,
        "dataset_manifest": dataset_manifest,
        "sample_manifest": sample_manifest,
        "prediction_plot": str(
            output_root / "predictions" / "LSTnet_prediction_comparison_with_guided_0.2.png"
        ),
        "mse_results": mse_results,
        "prediction_results_rate_0.2": prediction_results,
    }
    with open(output_root / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved full 2014 experiment outputs to: {output_root}")


if __name__ == "__main__":
    main()
