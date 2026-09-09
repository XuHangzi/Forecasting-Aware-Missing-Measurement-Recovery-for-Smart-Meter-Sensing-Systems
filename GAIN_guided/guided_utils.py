from __future__ import annotations

import importlib.util
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


GUIDED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUIDED_DIR.parent
BASE_GAIN_DIR = PROJECT_ROOT / "GAIN_pytorch"

if str(BASE_GAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_GAIN_DIR))

from utils import normalization, rmse_loss


@dataclass
class UnifiedElectricityData:
    timestamps: np.ndarray
    user_columns: list[str]
    raw_values: np.ndarray
    miss_data_x: np.ndarray
    data_m: np.ndarray
    gain_norm_values: np.ndarray
    gain_norm_parameters: dict
    train_pairs: np.ndarray
    val_pairs: np.ndarray
    test_pairs: np.ndarray


def _build_window_pairs(
    data_m: np.ndarray,
    lookback: int,
    horizon: int,
    test_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_timesteps, n_users = data_m.shape
    n_samples = n_timesteps - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("The series is too short to create LSTM windows.")

    test_size = int(n_samples * test_ratio)
    train_val_size = n_samples - test_size
    val_size = int(train_val_size * val_ratio)
    train_size = train_val_size - val_size

    train_pairs: list[tuple[int, int]] = []
    val_pairs: list[tuple[int, int]] = []
    test_pairs: list[tuple[int, int]] = []
    lookback_kernel = np.ones(lookback, dtype=np.float32)

    for user_idx in range(n_users):
        missing_counts = np.convolve(
            (1 - data_m[:, user_idx]).astype(np.float32),
            lookback_kernel,
            mode="valid",
        )[:n_samples]
        valid_starts = np.nonzero(missing_counts > 0)[0]

        train_pairs.extend((user_idx, int(start)) for start in valid_starts[valid_starts < train_size])
        val_pairs.extend(
            (user_idx, int(start))
            for start in valid_starts[(valid_starts >= train_size) & (valid_starts < train_val_size)]
        )
        test_pairs.extend(
            (user_idx, int(start)) for start in valid_starts[valid_starts >= train_val_size]
        )

    return (
        np.asarray(train_pairs, dtype=np.int64),
        np.asarray(val_pairs, dtype=np.int64),
        np.asarray(test_pairs, dtype=np.int64),
    )


def load_unified_electricity_data(
    source_path: Path,
    miss_rate: float,
    lookback: int,
    horizon: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> UnifiedElectricityData:
    df = pd.read_csv(source_path)
    if "hour" not in df.columns:
        raise ValueError(f"Expected a time column named 'hour' in {source_path}.")

    timestamps = df["hour"].astype(str).to_numpy()
    user_columns = [col for col in df.columns if col != "hour"]
    raw_values = df[user_columns].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(seed)
    data_m = (rng.uniform(size=raw_values.shape) < (1 - miss_rate)).astype(np.float32)
    miss_data_x = raw_values.copy()
    miss_data_x[data_m == 0] = np.nan

    norm_data, norm_parameters = normalization(miss_data_x)
    gain_norm_values = np.nan_to_num(norm_data, nan=0.0).astype(np.float32)

    train_pairs, val_pairs, test_pairs = _build_window_pairs(
        data_m=data_m,
        lookback=lookback,
        horizon=horizon,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )

    if len(train_pairs) == 0 or len(val_pairs) == 0 or len(test_pairs) == 0:
        raise ValueError(
            "At least one of the train/val/test guided window splits is empty. "
            "Try a different missing rate or a longer time series."
        )

    return UnifiedElectricityData(
        timestamps=timestamps,
        user_columns=user_columns,
        raw_values=raw_values,
        miss_data_x=miss_data_x,
        data_m=data_m,
        gain_norm_values=gain_norm_values,
        gain_norm_parameters=norm_parameters,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
    )


def load_lstnet_module(lstnet_script_path: Path):
    spec = importlib.util.spec_from_file_location("guided_lstnet_module", lstnet_script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LSTNet module from {lstnet_script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenLSTNetPredictor:
    def __init__(
        self,
        lstnet_script_path: Path,
        lstnet_model_path: Path,
        scaler_path: Path,
        timestamps: Sequence[str],
        user_columns: Sequence[str],
        raw_values: np.ndarray,
        device: torch.device,
    ) -> None:
        self.device = device
        self.lstnet_module = load_lstnet_module(lstnet_script_path)
        self.lookback = int(self.lstnet_module.config.lookback)
        self.horizon = int(self.lstnet_module.config.horizon)

        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        if len(scalers) != len(user_columns):
            raise ValueError(
                f"The frozen LSTM scaler count ({len(scalers)}) does not match the data user count "
                f"({len(user_columns)})."
            )

        self.user_min = torch.tensor(
            [float(scaler.data_min_[0]) for scaler in scalers],
            dtype=torch.float32,
            device=device,
        )
        self.user_range = torch.tensor(
            [float(scaler.data_max_[0] - scaler.data_min_[0]) for scaler in scalers],
            dtype=torch.float32,
            device=device,
        )

        self.time_features = torch.tensor(
            self.lstnet_module.create_time_features(np.asarray(timestamps)),
            dtype=torch.float32,
            device=device,
        )
        self.raw_values = torch.tensor(raw_values, dtype=torch.float32, device=device)

        self.stat_kernel = torch.ones(1, 1, self.lookback, dtype=torch.float32, device=device)
        self.lstm_model = self.lstnet_module.LSTNetOptimized(n_users=len(user_columns)).to(device)
        state_dict = torch.load(lstnet_model_path, map_location=device, weights_only=True)
        self.lstm_model.load_state_dict(state_dict)
        self.lstm_model.eval()
        for param in self.lstm_model.parameters():
            param.requires_grad_(False)

    def normalize_for_lstm(self, raw_values: torch.Tensor) -> torch.Tensor:
        return (raw_values - self.user_min.unsqueeze(0)) / (self.user_range.unsqueeze(0) + 1e-6)

    def denormalize_predictions(
        self, pred_norm: torch.Tensor, user_indices: torch.Tensor
    ) -> torch.Tensor:
        user_min = self.user_min[user_indices].unsqueeze(1)
        user_range = self.user_range[user_indices].unsqueeze(1)
        return pred_norm * (user_range + 1e-6) + user_min

    def build_stat_features(self, normalized_values: torch.Tensor) -> torch.Tensor:
        series = normalized_values.transpose(0, 1).unsqueeze(1)
        padded_series = F.pad(series, (self.lookback - 1, 0), value=0.0)
        padded_ones = F.pad(torch.ones_like(series), (self.lookback - 1, 0), value=0.0)

        rolling_sum = F.conv1d(padded_series, self.stat_kernel)
        rolling_count = F.conv1d(padded_ones, self.stat_kernel)
        rolling_sum_sq = F.conv1d(padded_series * padded_series, self.stat_kernel)
        rolling_max = F.max_pool1d(
            F.pad(series, (self.lookback - 1, 0), value=-1e9),
            kernel_size=self.lookback,
            stride=1,
        )

        rolling_mean = rolling_sum / rolling_count
        rolling_var = torch.clamp(rolling_sum_sq / rolling_count - rolling_mean.square(), min=0.0)
        rolling_std = torch.sqrt(rolling_var + 1e-8)
        peak_ratio = rolling_max / (rolling_mean + 1e-8)

        stat_features = torch.cat([rolling_mean, rolling_std, rolling_max, peak_ratio], dim=1)
        return stat_features.permute(2, 0, 1)

    def _build_lstm_batch(
        self,
        normalized_values: torch.Tensor,
        stat_features: torch.Tensor,
        pair_batch: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_batch = []
        y_batch = []
        user_idx_batch = []

        for user_idx, start_idx in pair_batch:
            user_idx = int(user_idx)
            start_idx = int(start_idx)
            end_idx = start_idx + self.lookback
            target_end_idx = end_idx + self.horizon

            elec_feat = normalized_values[start_idx:end_idx, user_idx].unsqueeze(-1)
            time_feat = self.time_features[start_idx:end_idx]
            stat_feat = stat_features[start_idx:end_idx, user_idx, :]
            x_batch.append(torch.cat([elec_feat, time_feat, stat_feat], dim=1))
            y_batch.append(self.raw_values[end_idx:target_end_idx, user_idx])
            user_idx_batch.append(user_idx)

        return (
            torch.stack(x_batch, dim=0),
            torch.tensor(user_idx_batch, dtype=torch.long, device=self.device),
            torch.stack(y_batch, dim=0),
        )

    def prediction_loss_from_raw(
        self, raw_imputed_values: torch.Tensor, pair_batch: np.ndarray
    ) -> torch.Tensor:
        normalized_values = self.normalize_for_lstm(raw_imputed_values)
        stat_features = self.build_stat_features(normalized_values)
        x_batch, user_indices, y_true_raw = self._build_lstm_batch(
            normalized_values, stat_features, pair_batch
        )

        with torch.backends.cudnn.flags(enabled=False):
            pred_norm = self.lstm_model(x_batch, user_indices).squeeze(-1)
        pred_raw = self.denormalize_predictions(pred_norm, user_indices)
        return torch.mean((pred_raw - y_true_raw) ** 2)

    def evaluate_prediction_mse(
        self,
        raw_imputed_values: torch.Tensor,
        pair_array: np.ndarray,
        batch_size: int,
    ) -> float:
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for start in range(0, len(pair_array), batch_size):
                pair_batch = pair_array[start : start + batch_size]
                batch_loss = self.prediction_loss_from_raw(raw_imputed_values, pair_batch)
                total_loss += batch_loss.item() * len(pair_batch)
                total_count += len(pair_batch)

        return total_loss / max(total_count, 1)


def compute_imputation_rmse(
    ori_data_x: np.ndarray, imputed_data_x: np.ndarray, data_m: np.ndarray
) -> float:
    return float(rmse_loss(ori_data_x, imputed_data_x, data_m))
