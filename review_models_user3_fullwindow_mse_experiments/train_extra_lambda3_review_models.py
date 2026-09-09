# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
REVIEW_DIR = PROJECT_ROOT / "review_training_hyperparam_experiments"

if str(REVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(REVIEW_DIR))

import run_review_experiments as review  # noqa: E402


EXTRA_LAMBDA3_VALUES = [1e-5, 1e-1]


def lambda_model_dir(value: float) -> Path:
    return review.MODELS_DIR / f"lambda_{value:g}".replace(".", "p")


def main() -> None:
    review.prepare_output_dirs()
    review.set_seed(review.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_path = review.GAIN_DIR / "data" / "electricity_data_2013.csv"
    source_df = pd.read_csv(source_path)
    predictor = review.FrozenLSTNetPredictor(
        lstnet_script_path=review.LSTM_DIR / "LSTNet_revised.py",
        lstnet_model_path=review.LSTM_DIR / "lstnet_best_model_optimized.pth",
        scaler_path=review.LSTM_DIR / "lstnet_scalers_optimized.pkl",
        timestamps=source_df["hour"].astype(str).to_numpy(),
        user_columns=[col for col in source_df.columns if col != "hour"],
        raw_values=source_df.drop(columns=["hour"]).to_numpy(dtype="float32"),
        device=device,
    )
    dataset = review.load_unified_electricity_data(
        source_path=source_path,
        miss_rate=review.MISS_RATE,
        lookback=predictor.lookback,
        horizon=predictor.horizon,
        test_ratio=0.2,
        val_ratio=0.1,
        seed=review.SEED,
    )
    val_pairs = review.sample_pairs(dataset.val_pairs, review.EVAL_PAIR_LIMIT, review.SEED + 1)
    test_pairs = review.sample_pairs(dataset.test_pairs, review.EVAL_PAIR_LIMIT, review.SEED + 2)

    baseline_path = review.MODELS_DIR / "time_gain" / "generator.pth"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline review generator: {baseline_path}")
    baseline_state = torch.load(baseline_path, map_location=device, weights_only=True)

    for value in EXTRA_LAMBDA3_VALUES:
        model_dir = lambda_model_dir(value)
        if (model_dir / "guided_generator_best.pth").exists() and (
            model_dir / "guided_imputed_processed.csv"
        ).exists():
            print(f"Reuse existing extra lambda_3 model: {value:g}")
            continue

        print(f"Training extra lambda_3 model: {value:g}")
        review.train_guided_gain(
            dataset=dataset,
            predictor=predictor,
            baseline_state=baseline_state,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            config=review.GuidedConfig(
                name="Fine-tuned GAIN",
                iterations=review.SWEEP_ITERATIONS,
                alpha=review.DEFAULT_ALPHA,
                prediction_weight=value,
                model_dir=model_dir,
            ),
            device=device,
        )

    print("Extra lambda_3 model training completed.")


if __name__ == "__main__":
    main()
