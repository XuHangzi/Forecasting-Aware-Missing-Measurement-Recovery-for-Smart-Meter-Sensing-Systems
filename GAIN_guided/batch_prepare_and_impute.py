# coding=utf-8

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


GUIDED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUIDED_DIR.parent
BASE_GAIN_DIR = PROJECT_ROOT / "GAIN_pytorch"

if str(BASE_GAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_GAIN_DIR))

from gain import Generator
from utils import normalization, renormalization


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def find_2014_reference(reference_dir: Path) -> Path:
    candidates = sorted(reference_dir.glob("electricity_data_2014*.csv"))
    for path in candidates:
        if "原版" in path.stem:
            return path
    if not candidates:
        raise FileNotFoundError(f"No 2014 reference CSV was found under {reference_dir}")
    return candidates[0]


def load_guided_generator(weight_path: Path, n_users: int, device: torch.device) -> Generator:
    model = Generator(n_users, n_users).to(device)
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load guided generator from {weight_path}. "
            f"The checkpoint likely does not match the current user count ({n_users}). "
            f"Please retrain guided GAIN after switching datasets."
        ) from e
    model.eval()
    return model


def infer_missing_mask(
    reference_values: np.ndarray,
    incomplete_values: np.ndarray,
) -> np.ndarray:
    nan_mask = np.isnan(incomplete_values)
    zero_missing_mask = (incomplete_values == 0) & (reference_values != 0)
    return nan_mask | zero_missing_mask


def guided_impute_array(
    generator: Generator,
    incomplete_values: np.ndarray,
    missing_mask: np.ndarray,
    device: torch.device,
    apply_rounding: bool,
) -> np.ndarray:
    miss_data = incomplete_values.astype(np.float32).copy()
    miss_data[missing_mask] = np.nan

    norm_data, norm_parameters = normalization(miss_data)
    observed_gain_norm = np.nan_to_num(norm_data, nan=0.0).astype(np.float32)
    data_m = (~np.isnan(miss_data)).astype(np.float32)

    observed_tensor = torch.tensor(observed_gain_norm, dtype=torch.float32, device=device)
    data_m_tensor = torch.tensor(data_m, dtype=torch.float32, device=device)
    noise_tensor = torch.rand_like(observed_tensor) * 0.01

    full_x = data_m_tensor * observed_tensor + (1 - data_m_tensor) * noise_tensor
    with torch.no_grad():
        generated = generator(full_x, data_m_tensor).cpu().numpy()

    imputed_norm = data_m * observed_gain_norm + (1 - data_m) * generated
    imputed_raw = renormalization(imputed_norm, norm_parameters)
    if apply_rounding:
        from utils import rounding

        imputed_raw = rounding(imputed_raw, miss_data)
    return imputed_raw.astype(np.float32)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dir = resolve_path(args.input_dir, GUIDED_DIR)
    reference_dir = resolve_path(args.reference_dir, GUIDED_DIR)
    generator_path = resolve_path(args.guided_generator, GUIDED_DIR)
    output_root = resolve_path(args.output_root, GUIDED_DIR)

    prepared_dir = output_root / "prepared_4users"
    imputed_dir = output_root / "guided_imputed_outputs"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    imputed_dir.mkdir(parents=True, exist_ok=True)

    reference_path = find_2014_reference(reference_dir)
    reference_df = pd.read_csv(reference_path)
    time_column = reference_df.columns[0]
    user_columns = list(reference_df.columns[1:5])
    selected_columns = [time_column] + user_columns
    reference_selected = reference_df[selected_columns].copy()
    reference_selected.to_csv(output_root / "electricity_data_2014_reference_4users.csv", index=False)

    generator = load_guided_generator(generator_path, len(user_columns), device)

    summary = {
        "reference_file": str(reference_path),
        "selected_columns": selected_columns,
        "processed_files": [],
    }

    input_files = sorted(input_dir.glob("*.csv"))
    if not input_files:
        raise FileNotFoundError(f"No CSV files were found under {input_dir}")

    reference_values = reference_selected[user_columns].to_numpy(dtype=np.float32)

    for input_path in input_files:
        input_df = pd.read_csv(input_path)
        missing_df_selected = input_df[selected_columns].copy()
        prepared_path = prepared_dir / input_path.name
        missing_df_selected.to_csv(prepared_path, index=False)

        incomplete_values = missing_df_selected[user_columns].to_numpy(dtype=np.float32)
        missing_mask = infer_missing_mask(reference_values, incomplete_values)
        imputed_values = guided_impute_array(
            generator,
            incomplete_values,
            missing_mask,
            device,
            apply_rounding=args.apply_rounding,
        )

        imputed_df = pd.DataFrame(imputed_values, columns=user_columns)
        imputed_df.insert(0, time_column, missing_df_selected[time_column].to_numpy())

        imputed_with_time_path = imputed_dir / f"{input_path.stem}_guided_imputed.csv"
        imputed_processed_path = imputed_dir / f"{input_path.stem}_guided_imputed_processed.csv"
        imputed_df.to_csv(imputed_with_time_path, index=False)
        imputed_df[user_columns].to_csv(imputed_processed_path, index=False)

        file_summary = {
            "input_file": str(input_path),
            "prepared_file": str(prepared_path),
            "imputed_with_time_file": str(imputed_with_time_path),
            "imputed_processed_file": str(imputed_processed_path),
            "missing_count_in_4users": int(missing_mask.sum()),
        }
        summary["processed_files"].append(file_summary)
        print(
            f"Processed {input_path.name} | "
            f"4-user missing count={file_summary['missing_count_in_4users']}"
        )

    with open(output_root / "batch_imputation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved prepared 4-user files to: {prepared_dir}")
    print(f"Saved guided imputed outputs to: {imputed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="../guided_gain_imputed",
        type=str,
        help="Directory containing the missing-data CSV files to process.",
    )
    parser.add_argument(
        "--reference_dir",
        default="../GAIN_pytorch/data",
        type=str,
        help="Directory containing the original 2014 reference CSV.",
    )
    parser.add_argument(
        "--guided_generator",
        default="saved_models_guided/guided_generator_best.pth",
        type=str,
        help="Path to the trained guided GAIN generator checkpoint.",
    )
    parser.add_argument(
        "--output_root",
        default="guided_gain_imputed_4users",
        type=str,
        help="Root directory for prepared 4-user files and imputed outputs.",
    )
    parser.add_argument(
        "--apply_rounding",
        action="store_true",
        help="Apply GAIN's categorical rounding heuristic after imputation.",
    )

    main(parser.parse_args())
