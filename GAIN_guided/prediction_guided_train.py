# coding=utf-8

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch


GUIDED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUIDED_DIR.parent
BASE_GAIN_DIR = PROJECT_ROOT / "GAIN_pytorch"

if str(GUIDED_DIR) not in sys.path:
    sys.path.insert(0, str(GUIDED_DIR))
if str(BASE_GAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_GAIN_DIR))

from gain import Discriminator, Generator, xavier_init_
from guided_utils import (
    FrozenLSTNetPredictor,
    compute_imputation_rmse,
    load_unified_electricity_data,
)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_pair_batch(pair_array: np.ndarray, batch_size: int) -> np.ndarray:
    if len(pair_array) <= batch_size:
        return pair_array

    indices = np.random.choice(len(pair_array), size=batch_size, replace=False)
    return pair_array[indices]


def build_full_imputed_raw(
    generator: Generator,
    observed_gain_norm: torch.Tensor,
    data_m: torch.Tensor,
    guidance_noise: torch.Tensor,
    gain_min: torch.Tensor,
    gain_range: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_x = data_m * observed_gain_norm + (1 - data_m) * guidance_noise
    generated = generator(full_x, data_m)
    imputed_gain_norm = data_m * observed_gain_norm + (1 - data_m) * generated
    imputed_raw = imputed_gain_norm * (gain_range + 1e-6) + gain_min
    return imputed_raw, imputed_gain_norm


def load_generator_weights(generator: Generator, weight_path: Path, device: torch.device) -> None:
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    try:
        generator.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load baseline GAIN generator from {weight_path}. "
            "The checkpoint likely comes from a different user-count experiment. "
            "Please retrain baseline GAIN on the current dataset before training guided GAIN."
        ) from e


def save_imputed_outputs(
    model_dir: Path,
    timestamps: np.ndarray,
    user_columns: list[str],
    imputed_raw: np.ndarray,
) -> None:
    df = pd.DataFrame(imputed_raw, columns=user_columns)
    df.insert(0, "hour", timestamps)
    df.to_csv(model_dir / "guided_imputed_with_time.csv", index=False)
    df[user_columns].to_csv(model_dir / "guided_imputed_processed.csv", index=False)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_path = resolve_path(args.source_data, GUIDED_DIR)
    pretrained_generator_path = resolve_path(args.pretrained_generator, GUIDED_DIR)
    lstm_script_path = resolve_path(args.lstm_script, GUIDED_DIR)
    lstm_model_path = resolve_path(args.lstm_model, GUIDED_DIR)
    lstm_scalers_path = resolve_path(args.lstm_scalers, GUIDED_DIR)
    model_dir = resolve_path(args.model_dir, GUIDED_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(source_path)
    predictor = FrozenLSTNetPredictor(
        lstnet_script_path=lstm_script_path,
        lstnet_model_path=lstm_model_path,
        scaler_path=lstm_scalers_path,
        timestamps=source_df["hour"].astype(str).to_numpy(),
        user_columns=[col for col in source_df.columns if col != "hour"],
        raw_values=source_df.drop(columns=["hour"]).to_numpy(dtype=np.float32),
        device=device,
    )

    dataset = load_unified_electricity_data(
        source_path=source_path,
        miss_rate=args.miss_rate,
        lookback=predictor.lookback,
        horizon=predictor.horizon,
        test_ratio=0.2,
        val_ratio=0.1,
        seed=args.seed,
    )

    n_rows, n_users = dataset.raw_values.shape
    print(f"Unified source data: rows={n_rows}, users={n_users}")
    print(
        f"Guided windows with missing values: train={len(dataset.train_pairs)}, "
        f"val={len(dataset.val_pairs)}, test={len(dataset.test_pairs)}"
    )

    observed_gain_norm = torch.tensor(dataset.gain_norm_values, dtype=torch.float32, device=device)
    data_m = torch.tensor(dataset.data_m, dtype=torch.float32, device=device)
    gain_min = torch.tensor(
        dataset.gain_norm_parameters["min_val"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    gain_range = torch.tensor(
        dataset.gain_norm_parameters["max_val"], dtype=torch.float32, device=device
    ).unsqueeze(0)

    guidance_noise = torch.rand_like(observed_gain_norm) * 0.01

    baseline_generator = Generator(n_users, n_users).to(device)
    guided_generator = Generator(n_users, n_users).to(device)
    load_generator_weights(baseline_generator, pretrained_generator_path, device)
    load_generator_weights(guided_generator, pretrained_generator_path, device)
    baseline_generator.eval()

    discriminator = Discriminator(n_users, n_users).to(device)
    discriminator.apply(xavier_init_)

    g_optimizer = torch.optim.Adam(guided_generator.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    with torch.no_grad():
        baseline_raw, _ = build_full_imputed_raw(
            baseline_generator,
            observed_gain_norm,
            data_m,
            guidance_noise,
            gain_min,
            gain_range,
        )
        baseline_val_mse = predictor.evaluate_prediction_mse(
            baseline_raw, dataset.val_pairs, args.eval_batch_size
        )
        baseline_test_mse = predictor.evaluate_prediction_mse(
            baseline_raw, dataset.test_pairs, args.eval_batch_size
        )
        baseline_imputation_rmse = compute_imputation_rmse(
            dataset.raw_values, baseline_raw.cpu().numpy(), dataset.data_m
        )

    print(
        "Baseline frozen-LSTM prediction metrics | "
        f"val_mse={baseline_val_mse:.6f} | test_mse={baseline_test_mse:.6f} | "
        f"imputation_rmse={baseline_imputation_rmse:.6f}"
    )

    best_state = deepcopy(guided_generator.state_dict())
    best_metrics = {
        "val_prediction_mse": baseline_val_mse,
        "test_prediction_mse": baseline_test_mse,
        "imputation_rmse": baseline_imputation_rmse,
        "iteration": 0,
        "improved_over_baseline": False,
    }

    for iteration in range(1, args.iterations + 1):
        row_indices = torch.randint(0, n_rows, (args.batch_size,), device=device)
        x_mb = observed_gain_norm[row_indices]
        m_mb = data_m[row_indices]
        z_mb = torch.rand((args.batch_size, n_users), device=device) * 0.01
        h_mb = m_mb * (torch.rand((args.batch_size, n_users), device=device) < args.hint_rate).float()
        x_tilde = m_mb * x_mb + (1 - m_mb) * z_mb

        d_optimizer.zero_grad()
        with torch.no_grad():
            g_sample_for_d = guided_generator(x_tilde, m_mb)
        hat_x_for_d = x_tilde * m_mb + g_sample_for_d * (1 - m_mb)
        d_prob = discriminator(hat_x_for_d, h_mb)
        d_loss = -torch.mean(
            m_mb * torch.log(d_prob + 1e-8)
            + (1 - m_mb) * torch.log(1 - d_prob + 1e-8)
        )
        d_loss.backward()
        d_optimizer.step()

        row_indices = torch.randint(0, n_rows, (args.batch_size,), device=device)
        x_mb = observed_gain_norm[row_indices]
        m_mb = data_m[row_indices]
        z_mb = torch.rand((args.batch_size, n_users), device=device) * 0.01
        h_mb = m_mb * (torch.rand((args.batch_size, n_users), device=device) < args.hint_rate).float()
        x_tilde = m_mb * x_mb + (1 - m_mb) * z_mb

        g_optimizer.zero_grad()
        g_sample = guided_generator(x_tilde, m_mb)
        hat_x = x_tilde * m_mb + g_sample * (1 - m_mb)
        d_prob = discriminator(hat_x, h_mb)

        g_loss_adv = -torch.mean((1 - m_mb) * torch.log(d_prob + 1e-8))
        g_loss_obs = torch.mean((m_mb * x_tilde - m_mb * g_sample) ** 2) / (
            torch.mean(m_mb) + 1e-8
        )

        full_imputed_raw, _ = build_full_imputed_raw(
            guided_generator,
            observed_gain_norm,
            data_m,
            guidance_noise,
            gain_min,
            gain_range,
        )
        pair_batch = sample_pair_batch(dataset.train_pairs, args.prediction_batch_size)
        prediction_loss = predictor.prediction_loss_from_raw(full_imputed_raw, pair_batch)

        if args.prediction_warmup_iters > 0:
            warmup_ratio = min(1.0, iteration / args.prediction_warmup_iters)
        else:
            warmup_ratio = 1.0
        prediction_weight = args.prediction_weight * warmup_ratio

        g_total_loss = g_loss_adv + args.alpha * g_loss_obs + prediction_weight * prediction_loss
        g_total_loss.backward()
        g_optimizer.step()

        if iteration % args.log_interval == 0 or iteration == 1:
            print(
                f"Iter {iteration:05d} | "
                f"D={d_loss.item():.6f} | "
                f"G_adv={g_loss_adv.item():.6f} | "
                f"G_obs={g_loss_obs.item():.6f} | "
                f"Pred={prediction_loss.item():.6f} | "
                f"Pred_w={prediction_weight:.6f}"
            )

        if iteration % args.eval_interval == 0 or iteration == args.iterations:
            with torch.no_grad():
                current_imputed_raw, _ = build_full_imputed_raw(
                    guided_generator,
                    observed_gain_norm,
                    data_m,
                    guidance_noise,
                    gain_min,
                    gain_range,
                )
                val_prediction_mse = predictor.evaluate_prediction_mse(
                    current_imputed_raw, dataset.val_pairs, args.eval_batch_size
                )
                test_prediction_mse = predictor.evaluate_prediction_mse(
                    current_imputed_raw, dataset.test_pairs, args.eval_batch_size
                )
                imputation_rmse = compute_imputation_rmse(
                    dataset.raw_values, current_imputed_raw.cpu().numpy(), dataset.data_m
                )

            print(
                f"Eval @ iter {iteration:05d} | "
                f"val_pred_mse={val_prediction_mse:.6f} | "
                f"test_pred_mse={test_prediction_mse:.6f} | "
                f"imputation_rmse={imputation_rmse:.6f}"
            )

            if val_prediction_mse < best_metrics["val_prediction_mse"]:
                best_state = deepcopy(guided_generator.state_dict())
                best_metrics = {
                    "val_prediction_mse": val_prediction_mse,
                    "test_prediction_mse": test_prediction_mse,
                    "imputation_rmse": imputation_rmse,
                    "iteration": iteration,
                    "improved_over_baseline": val_prediction_mse < baseline_val_mse,
                }
                print(
                    f"New best guided checkpoint at iter {iteration:05d} "
                    f"(val_pred_mse={val_prediction_mse:.6f})"
                )

    guided_generator.load_state_dict(best_state)
    guided_generator.eval()

    with torch.no_grad():
        best_imputed_raw, _ = build_full_imputed_raw(
            guided_generator,
            observed_gain_norm,
            data_m,
            guidance_noise,
            gain_min,
            gain_range,
        )

    torch.save(guided_generator.state_dict(), model_dir / "guided_generator_best.pth")
    np.save(model_dir / "guided_norm_parameters.npy", dataset.gain_norm_parameters)
    np.save(model_dir / "guided_mask.npy", dataset.data_m)
    save_imputed_outputs(
        model_dir=model_dir,
        timestamps=dataset.timestamps,
        user_columns=dataset.user_columns,
        imputed_raw=best_imputed_raw.cpu().numpy(),
    )

    metrics = {
        "baseline_val_prediction_mse": baseline_val_mse,
        "baseline_test_prediction_mse": baseline_test_mse,
        "baseline_imputation_rmse": baseline_imputation_rmse,
        "best_guided_val_prediction_mse": best_metrics["val_prediction_mse"],
        "best_guided_test_prediction_mse": best_metrics["test_prediction_mse"],
        "best_guided_imputation_rmse": best_metrics["imputation_rmse"],
        "best_iteration": best_metrics["iteration"],
        "improved_over_baseline": best_metrics["improved_over_baseline"],
        "train_pair_count": int(len(dataset.train_pairs)),
        "val_pair_count": int(len(dataset.val_pairs)),
        "test_pair_count": int(len(dataset.test_pairs)),
    }
    with open(model_dir / "guided_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nPrediction-guided GAIN training completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_data",
        default="../GAIN_pytorch/data/electricity_data_2013.csv",
        type=str,
        help="Path to the active raw electricity dataset with the hour column.",
    )
    parser.add_argument(
        "--pretrained_generator",
        default="../GAIN_pytorch/saved_models/generator.pth",
        type=str,
        help="Path to the baseline GAIN generator checkpoint.",
    )
    parser.add_argument(
        "--lstm_script",
        default="../LSTM/LSTNet_revised.py",
        type=str,
        help="Path to the frozen LSTM script definition.",
    )
    parser.add_argument(
        "--lstm_model",
        default="../LSTM/lstnet_best_model_optimized.pth",
        type=str,
        help="Path to the frozen LSTM checkpoint.",
    )
    parser.add_argument(
        "--lstm_scalers",
        default="../LSTM/lstnet_scalers_optimized.pkl",
        type=str,
        help="Path to the frozen LSTM scaler file.",
    )
    parser.add_argument("--model_dir", default="saved_models_guided", type=str)
    parser.add_argument("--miss_rate", default=0.2, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--prediction_batch_size", default=64, type=int)
    parser.add_argument("--hint_rate", default=0.9, type=float)
    parser.add_argument("--alpha", default=100.0, type=float)
    parser.add_argument("--prediction_weight", default=1e-3, type=float)
    parser.add_argument("--prediction_warmup_iters", default=500, type=int)
    parser.add_argument("--iterations", default=3000, type=int)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)

    main(parser.parse_args())
