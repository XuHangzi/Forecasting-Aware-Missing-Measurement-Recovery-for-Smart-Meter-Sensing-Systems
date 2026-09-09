# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import gc
import json
import random
import shutil
import sys
import time
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
GAIN_DIR = PROJECT_ROOT / "GAIN_pytorch"
GUIDED_DIR = PROJECT_ROOT / "GAIN_guided"
LSTM_DIR = PROJECT_ROOT / "LSTM"

for path in (GAIN_DIR, GUIDED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gain import Discriminator, Generator, xavier_init_  # noqa: E402
from guided_utils import (  # noqa: E402
    FrozenLSTNetPredictor,
    compute_imputation_rmse,
    load_unified_electricity_data,
)
from utils import (  # noqa: E402
    binary_sampler,
    normalization,
    renormalization,
    rounding,
    sample_batch_index,
    uniform_sampler,
)


# Reproducible budget for the final review supplement.
SEED = 42
MISS_RATE = 0.2
TIME_ITERATIONS = 3000
SWEEP_ITERATIONS = 3000
BATCH_SIZE = 64
PREDICTION_BATCH_SIZE = 64
EVAL_PAIR_LIMIT = 768
EVAL_INTERVAL = 75
HINT_RATE = 0.9
DEFAULT_ALPHA = 100.0
DEFAULT_PREDICTION_WEIGHT = 1e-3
LR = 1e-3
LAMBDA_VALUES = [0.0, 1e-4, 1e-3, 1e-2]
ALPHA_VALUES = [1.0, 10.0, 50.0, 100.0, 200.0, 500.0]

RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
MODELS_DIR = EXP_DIR / "models"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_cuda_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def sample_pairs(pair_array: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(pair_array) <= limit:
        return pair_array
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pair_array), size=limit, replace=False)
    return pair_array[np.sort(indices)]


def sample_pair_batch(pair_array: np.ndarray, batch_size: int) -> np.ndarray:
    if len(pair_array) <= batch_size:
        return pair_array
    indices = np.random.choice(len(pair_array), size=batch_size, replace=False)
    return pair_array[indices]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_with_formula_notation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted_rows: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key == "alpha":
                converted["lambda_2"] = value
            elif key == "prediction_weight":
                converted["lambda_3"] = value
            else:
                converted[key] = value
        converted_rows.append(converted)
    return converted_rows


def add_docx_paragraph(lines: list[str], text: str, bold: bool = False) -> None:
    escaped = escape(text)
    bold_xml = "<w:rPr><w:b/></w:rPr>" if bold else ""
    lines.append(
        "<w:p><w:r>"
        f"{bold_xml}<w:t xml:space=\"preserve\">{escaped}</w:t>"
        "</w:r></w:p>"
    )


def write_simple_docx(path: Path, paragraphs: list[tuple[str, bool]]) -> None:
    document_lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        (
            '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
            'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            'xmlns:o="urn:schemas-microsoft-com:office:office" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
            'xmlns:v="urn:schemas-microsoft-com:vml" '
            'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:w10="urn:schemas-microsoft-com:office:word" '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
            'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
            'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
            'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
            'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
            'mc:Ignorable="w14 wp14"><w:body>'
        ),
    ]
    for text, bold in paragraphs:
        add_docx_paragraph(document_lines, text, bold=bold)
    document_lines.append(
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
        "</w:sectPr></w:body></w:document>"
    )
    document_xml = "\n".join(document_lines)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Training time and hyperparameter experiment plan</dc:title>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("docProps/core.xml", core)
        docx.writestr("docProps/app.xml", app)
        docx.writestr("word/document.xml", document_xml)


def write_plan_docx() -> None:
    paragraphs = [
        ("训练时间与关键超参数影响实验方案", True),
        ("一、实验目的", True),
        ("针对评审意见，补充两个轻量但直接的实验：第一，分析普通 GAIN 与微调 GAIN 的训练时间差异；第二，分析关键超参数对预测性能和填补性能的影响。", False),
        ("本文采用公式 Loss_G2 = Loss_G + λ2 Loss_D2 + λ3 Loss_P 表示微调 GAIN 的生成器损失。其中 λ2 为可观测数据重构一致性损失权重，λ3 为 LSTM 预测损失权重。", False),
        ("二、实验设置", True),
        ("数据集使用当前四用户电力数据，缺失率固定为 0.2，硬件环境为 NVIDIA RTX 3050 6GB，Python 环境使用 conda 环境 mytorch。", False),
        ("为保证补实验与正式训练设置一致，训练时间对比和超参数敏感性实验均使用 3000 次迭代。", False),
        ("三、训练时间分析", True),
        ("比较对象包括普通 GAIN 和加入 LSTM 预测损失的微调 GAIN。记录总训练时间、每 100 次迭代耗时、峰值 GPU 显存、模型参数量、填补 RMSE 和基于冻结 LSTM 的预测 MSE。", False),
        ("该实验用于说明微调 GAIN 相比普通 GAIN 的额外训练代价，并判断预测性能提升是否值得该额外开销。", False),
        ("四、关键超参数影响", True),
        ("第一个关键超参数是 λ3，用于控制 LSTM 预测损失 Loss_P 在生成器损失中的权重。实验取值为 0、1e-4、1e-3、1e-2。", False),
        ("第二个关键超参数是 λ2，用于控制可观测位置的重构一致性损失 Loss_D2。实验取值为 1、10、50、100、200、500。", False),
        ("评估指标包括填补 RMSE、验证集预测 MSE、测试集预测 MSE 和训练时间。", False),
        ("五、输出内容", True),
        ("实验输出包括训练时间对比表、λ3 敏感性表、λ2 敏感性表，以及对应的 JPEG 图像。所有结果保存在 review_training_hyperparam_experiments 文件夹中。", False),
    ]
    try:
        write_simple_docx(EXP_DIR / "experiment_plan.docx", paragraphs)
    except PermissionError:
        write_simple_docx(EXP_DIR / "experiment_plan_latest.docx", paragraphs)


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


def train_baseline_gain(
    miss_data_x: np.ndarray,
    raw_values: np.ndarray,
    data_m: np.ndarray,
    predictor: FrozenLSTNetPredictor,
    val_pairs: np.ndarray,
    test_pairs: np.ndarray,
    device: torch.device,
    iterations: int,
    model_dir: Path,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    set_seed(SEED)
    model_dir.mkdir(parents=True, exist_ok=True)
    reset_cuda_peak()

    batch_size = BATCH_SIZE
    hint_rate = HINT_RATE
    no, dim = miss_data_x.shape
    h_dim = int(dim)

    norm_data, norm_parameters = normalization(miss_data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0.0).astype(np.float32)

    norm_tensor = torch.tensor(norm_data_x, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(data_m, dtype=torch.float32, device=device)

    generator = Generator(dim, h_dim).to(device)
    discriminator = Discriminator(dim, h_dim).to(device)
    generator.apply(xavier_init_)
    discriminator.apply(xavier_init_)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR)

    start_time = time.perf_counter()
    for _ in range(iterations):
        batch_idx = sample_batch_index(no, batch_size)
        x_mb = norm_tensor[batch_idx, :]
        m_mb = mask_tensor[batch_idx, :]
        z_mb = torch.tensor(
            uniform_sampler(0, 0.01, batch_size, dim),
            dtype=torch.float32,
            device=device,
        )
        h_mb_temp = torch.tensor(
            binary_sampler(hint_rate, batch_size, dim),
            dtype=torch.float32,
            device=device,
        )
        h_mb = m_mb * h_mb_temp
        x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

        d_optimizer.zero_grad()
        g_sample = generator(x_mb, m_mb)
        hat_x = x_mb * m_mb + g_sample * (1 - m_mb)
        d_prob = discriminator(hat_x, h_mb)
        d_loss = -torch.mean(
            m_mb * torch.log(d_prob + 1e-8)
            + (1 - m_mb) * torch.log(1 - d_prob + 1e-8)
        )
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        g_sample = generator(x_mb, m_mb)
        hat_x = x_mb * m_mb + g_sample * (1 - m_mb)
        d_prob = discriminator(hat_x, h_mb)
        g_loss_adv = -torch.mean((1 - m_mb) * torch.log(d_prob + 1e-8))
        g_loss_obs = torch.mean((m_mb * x_mb - m_mb * g_sample) ** 2) / (
            torch.mean(m_mb) + 1e-8
        )
        g_loss = g_loss_adv + alpha * g_loss_obs
        g_loss.backward()
        g_optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    z_full = torch.tensor(
        uniform_sampler(0, 0.01, no, dim),
        dtype=torch.float32,
        device=device,
    )
    x_full = mask_tensor * norm_tensor + (1 - mask_tensor) * z_full
    with torch.no_grad():
        imputed_norm = generator(x_full, mask_tensor).detach().cpu().numpy()
    imputed_norm = data_m * norm_data_x + (1 - data_m) * imputed_norm
    imputed_data = renormalization(imputed_norm, norm_parameters)
    imputed_data = rounding(imputed_data, raw_values)

    imputed_tensor = torch.tensor(imputed_data, dtype=torch.float32, device=device)
    val_mse = predictor.evaluate_prediction_mse(imputed_tensor, val_pairs, EVAL_PAIR_LIMIT)
    test_mse = predictor.evaluate_prediction_mse(imputed_tensor, test_pairs, EVAL_PAIR_LIMIT)
    imputation_rmse = compute_imputation_rmse(raw_values, imputed_data, data_m)

    torch.save(generator.state_dict(), model_dir / "generator.pth")
    np.save(model_dir / "norm_parameters.npy", norm_parameters)

    metrics = {
        "model": "GAIN",
        "iterations": iterations,
        "alpha": alpha,
        "prediction_weight": 0.0,
        "training_time_sec": elapsed,
        "time_per_100_iter_sec": elapsed / iterations * 100,
        "peak_gpu_memory_mb": peak_cuda_mb(),
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "best_iteration": iterations,
        "imputation_rmse": imputation_rmse,
        "val_prediction_mse": val_mse,
        "test_prediction_mse": test_mse,
        "model_dir": str(model_dir),
    }
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    baseline_state = deepcopy(generator.state_dict())
    del generator, discriminator, norm_tensor, mask_tensor, imputed_tensor
    cleanup_cuda()
    return metrics, baseline_state


@dataclass
class GuidedConfig:
    name: str
    iterations: int
    alpha: float
    prediction_weight: float
    model_dir: Path


def train_guided_gain(
    dataset: Any,
    predictor: FrozenLSTNetPredictor,
    baseline_state: dict[str, torch.Tensor],
    val_pairs: np.ndarray,
    test_pairs: np.ndarray,
    config: GuidedConfig,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(SEED)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    reset_cuda_peak()

    n_rows, n_users = dataset.raw_values.shape
    observed_gain_norm = torch.tensor(dataset.gain_norm_values, dtype=torch.float32, device=device)
    data_m = torch.tensor(dataset.data_m, dtype=torch.float32, device=device)
    gain_min = torch.tensor(
        dataset.gain_norm_parameters["min_val"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    gain_range = torch.tensor(
        dataset.gain_norm_parameters["max_val"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    guidance_noise = torch.rand_like(observed_gain_norm) * 0.01

    generator = Generator(n_users, n_users).to(device)
    generator.load_state_dict(deepcopy(baseline_state))
    discriminator = Discriminator(n_users, n_users).to(device)
    discriminator.apply(xavier_init_)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR)

    best_state = deepcopy(generator.state_dict())
    best_val_mse = float("inf")
    best_test_mse = float("inf")
    best_rmse = float("inf")
    best_iteration = 0
    warmup_iters = max(1, min(100, config.iterations // 3))

    start_time = time.perf_counter()
    for iteration in range(1, config.iterations + 1):
        row_indices = torch.randint(0, n_rows, (BATCH_SIZE,), device=device)
        x_mb = observed_gain_norm[row_indices]
        m_mb = data_m[row_indices]
        z_mb = torch.rand((BATCH_SIZE, n_users), device=device) * 0.01
        h_mb = m_mb * (torch.rand((BATCH_SIZE, n_users), device=device) < HINT_RATE).float()
        x_tilde = m_mb * x_mb + (1 - m_mb) * z_mb

        d_optimizer.zero_grad()
        with torch.no_grad():
            g_sample_for_d = generator(x_tilde, m_mb)
        hat_x_for_d = x_tilde * m_mb + g_sample_for_d * (1 - m_mb)
        d_prob = discriminator(hat_x_for_d, h_mb)
        d_loss = -torch.mean(
            m_mb * torch.log(d_prob + 1e-8)
            + (1 - m_mb) * torch.log(1 - d_prob + 1e-8)
        )
        d_loss.backward()
        d_optimizer.step()

        row_indices = torch.randint(0, n_rows, (BATCH_SIZE,), device=device)
        x_mb = observed_gain_norm[row_indices]
        m_mb = data_m[row_indices]
        z_mb = torch.rand((BATCH_SIZE, n_users), device=device) * 0.01
        h_mb = m_mb * (torch.rand((BATCH_SIZE, n_users), device=device) < HINT_RATE).float()
        x_tilde = m_mb * x_mb + (1 - m_mb) * z_mb

        g_optimizer.zero_grad()
        g_sample = generator(x_tilde, m_mb)
        hat_x = x_tilde * m_mb + g_sample * (1 - m_mb)
        d_prob = discriminator(hat_x, h_mb)

        g_loss_adv = -torch.mean((1 - m_mb) * torch.log(d_prob + 1e-8))
        g_loss_obs = torch.mean((m_mb * x_tilde - m_mb * g_sample) ** 2) / (
            torch.mean(m_mb) + 1e-8
        )

        full_imputed_raw, _ = build_full_imputed_raw(
            generator,
            observed_gain_norm,
            data_m,
            guidance_noise,
            gain_min,
            gain_range,
        )
        pair_batch = sample_pair_batch(dataset.train_pairs, PREDICTION_BATCH_SIZE)
        prediction_loss = predictor.prediction_loss_from_raw(full_imputed_raw, pair_batch)
        warmup_ratio = min(1.0, iteration / warmup_iters)
        prediction_weight = config.prediction_weight * warmup_ratio

        total_loss = g_loss_adv + config.alpha * g_loss_obs + prediction_weight * prediction_loss
        total_loss.backward()
        g_optimizer.step()

        should_eval = iteration == config.iterations or iteration % EVAL_INTERVAL == 0
        if should_eval:
            with torch.no_grad():
                current_raw, _ = build_full_imputed_raw(
                    generator,
                    observed_gain_norm,
                    data_m,
                    guidance_noise,
                    gain_min,
                    gain_range,
                )
                val_mse = predictor.evaluate_prediction_mse(current_raw, val_pairs, EVAL_PAIR_LIMIT)
                test_mse = predictor.evaluate_prediction_mse(current_raw, test_pairs, EVAL_PAIR_LIMIT)
                rmse = compute_imputation_rmse(
                    dataset.raw_values,
                    current_raw.detach().cpu().numpy(),
                    dataset.data_m,
                )

            if val_mse < best_val_mse:
                best_state = deepcopy(generator.state_dict())
                best_val_mse = float(val_mse)
                best_test_mse = float(test_mse)
                best_rmse = float(rmse)
                best_iteration = iteration

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    generator.load_state_dict(best_state)
    torch.save(generator.state_dict(), config.model_dir / "guided_generator_best.pth")
    with torch.no_grad():
        best_raw, _ = build_full_imputed_raw(
            generator,
            observed_gain_norm,
            data_m,
            guidance_noise,
            gain_min,
            gain_range,
        )
    pd.DataFrame(best_raw.detach().cpu().numpy(), columns=dataset.user_columns).to_csv(
        config.model_dir / "guided_imputed_processed.csv",
        index=False,
    )

    metrics = {
        "model": config.name,
        "iterations": config.iterations,
        "alpha": config.alpha,
        "prediction_weight": config.prediction_weight,
        "training_time_sec": elapsed,
        "time_per_100_iter_sec": elapsed / config.iterations * 100,
        "peak_gpu_memory_mb": peak_cuda_mb(),
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "best_iteration": best_iteration,
        "imputation_rmse": best_rmse,
        "val_prediction_mse": best_val_mse,
        "test_prediction_mse": best_test_mse,
        "model_dir": str(config.model_dir),
    }
    with (config.model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    del generator, discriminator, observed_gain_norm, data_m, best_raw
    cleanup_cuda()
    return metrics


def plot_training_time(rows: list[dict[str, Any]]) -> None:
    labels = [row["model"] for row in rows]
    values = [row["training_time_sec"] / 60 for row in rows]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=["#4f81bd", "#c0504d"])
    plt.ylabel("Training time (min)")
    plt.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, rows):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['time_per_100_iter_sec']:.1f}s/100 iters",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_time_comparison.jpeg", dpi=300)
    plt.close()


def plot_lambda(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['prediction_weight']:.0e}" if row["prediction_weight"] else "0" for row in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax2 = ax1.twinx()
    line1 = ax1.plot(
        x,
        [row["test_prediction_mse"] for row in rows],
        marker="o",
        color="#4f81bd",
        label="Prediction MSE",
    )
    line2 = ax2.plot(
        x,
        [row["imputation_rmse"] for row in rows],
        marker="s",
        color="#c0504d",
        label="Imputation RMSE",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("λ3")
    ax1.set_ylabel("Prediction MSE")
    ax2.set_ylabel("Imputation RMSE")
    pred_values = [row["test_prediction_mse"] for row in rows]
    rmse_values = [row["imputation_rmse"] for row in rows]
    pred_range = max(pred_values) - min(pred_values)
    rmse_range = max(rmse_values) - min(rmse_values)
    ax1.set_ylim(
        min(pred_values) - max(pred_range * 0.05, 1e-6),
        max(pred_values) + max(pred_range * 0.35, 1e-6),
    )
    ax2.set_ylim(
        min(rmse_values) - max(rmse_range * 0.05, 1e-6),
        max(rmse_values) + max(rmse_range * 0.35, 1e-6),
    )
    ax1.grid(alpha=0.25)
    lines = line1 + line2
    ax1.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "lambda_3_sensitivity.jpeg", dpi=300)
    plt.close()


def plot_alpha(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['alpha']:.0f}" for row in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax2 = ax1.twinx()
    line1 = ax1.plot(
        x,
        [row["test_prediction_mse"] for row in rows],
        marker="o",
        color="#4f81bd",
        label="Prediction MSE",
    )
    line2 = ax2.plot(
        x,
        [row["imputation_rmse"] for row in rows],
        marker="s",
        color="#c0504d",
        label="Imputation RMSE",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("λ2")
    ax1.set_ylabel("Prediction MSE")
    ax2.set_ylabel("Imputation RMSE")
    ax1.grid(alpha=0.25)
    lines = line1 + line2
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "lambda_2_sensitivity.jpeg", dpi=300)
    plt.close()


def write_result_docx(all_results: dict[str, Any]) -> None:
    training = all_results["training_time"]
    lambda_rows = all_results["lambda_sensitivity"]
    alpha_rows = all_results["alpha_sensitivity"]

    paragraphs: list[tuple[str, bool]] = [
        ("训练时间与关键超参数实验结果汇总", True),
        ("一、训练时间分析", True),
    ]
    for row in training:
        paragraphs.append(
            (
                (
                    f"{row['model']}: 总训练时间 {row['training_time_sec']:.2f} 秒，"
                    f"每 100 次迭代 {row['time_per_100_iter_sec']:.2f} 秒，"
                    f"测试预测 MSE {row['test_prediction_mse']:.6f}，"
                    f"填补 RMSE {row['imputation_rmse']:.6f}。"
                ),
                False,
            )
        )

    paragraphs.append(("二、λ3 敏感性", True))
    for row in lambda_rows:
        paragraphs.append(
            (
                (
                    f"λ3={row['prediction_weight']}: "
                    f"测试预测 MSE={row['test_prediction_mse']:.6f}, "
                    f"填补 RMSE={row['imputation_rmse']:.6f}, "
                    f"训练时间={row['training_time_sec']:.2f} 秒。"
                ),
                False,
            )
        )

    paragraphs.append(("三、λ2 敏感性", True))
    for row in alpha_rows:
        paragraphs.append(
            (
                (
                    f"λ2={row['alpha']}: "
                    f"测试预测 MSE={row['test_prediction_mse']:.6f}, "
                    f"填补 RMSE={row['imputation_rmse']:.6f}, "
                    f"训练时间={row['training_time_sec']:.2f} 秒。"
                ),
                False,
            )
        )
    try:
        write_simple_docx(EXP_DIR / "experiment_results_summary.docx", paragraphs)
    except PermissionError:
        write_simple_docx(EXP_DIR / "experiment_results_summary_latest.docx", paragraphs)


def main() -> None:
    prepare_output_dirs()
    write_plan_docx()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_path = GAIN_DIR / "data" / "electricity_data_2013.csv"
    source_df = pd.read_csv(source_path)
    predictor = FrozenLSTNetPredictor(
        lstnet_script_path=LSTM_DIR / "LSTNet_revised.py",
        lstnet_model_path=LSTM_DIR / "lstnet_best_model_optimized.pth",
        scaler_path=LSTM_DIR / "lstnet_scalers_optimized.pkl",
        timestamps=source_df["hour"].astype(str).to_numpy(),
        user_columns=[col for col in source_df.columns if col != "hour"],
        raw_values=source_df.drop(columns=["hour"]).to_numpy(dtype=np.float32),
        device=device,
    )
    dataset = load_unified_electricity_data(
        source_path=source_path,
        miss_rate=MISS_RATE,
        lookback=predictor.lookback,
        horizon=predictor.horizon,
        test_ratio=0.2,
        val_ratio=0.1,
        seed=SEED,
    )
    val_pairs = sample_pairs(dataset.val_pairs, EVAL_PAIR_LIMIT, SEED + 1)
    test_pairs = sample_pairs(dataset.test_pairs, EVAL_PAIR_LIMIT, SEED + 2)

    print("Running training-time experiment: baseline GAIN")
    gain_metrics, baseline_state = train_baseline_gain(
        miss_data_x=dataset.miss_data_x,
        raw_values=dataset.raw_values,
        data_m=dataset.data_m,
        predictor=predictor,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        device=device,
        iterations=TIME_ITERATIONS,
        model_dir=MODELS_DIR / "time_gain",
        alpha=DEFAULT_ALPHA,
    )

    print("Running training-time experiment: fine-tuned GAIN")
    guided_time_metrics = train_guided_gain(
        dataset=dataset,
        predictor=predictor,
        baseline_state=baseline_state,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        config=GuidedConfig(
            name="Fine-tuned GAIN",
            iterations=TIME_ITERATIONS,
            alpha=DEFAULT_ALPHA,
            prediction_weight=DEFAULT_PREDICTION_WEIGHT,
            model_dir=MODELS_DIR / "time_fine_tuned_gain",
        ),
        device=device,
    )

    lambda_rows: list[dict[str, Any]] = []
    for value in LAMBDA_VALUES:
        print(f"Running prediction_weight sensitivity: {value}")
        lambda_rows.append(
            train_guided_gain(
                dataset=dataset,
                predictor=predictor,
                baseline_state=baseline_state,
                val_pairs=val_pairs,
                test_pairs=test_pairs,
                config=GuidedConfig(
                    name="Fine-tuned GAIN",
                    iterations=SWEEP_ITERATIONS,
                    alpha=DEFAULT_ALPHA,
                    prediction_weight=value,
                    model_dir=MODELS_DIR / f"lambda_{value:g}".replace(".", "p"),
                ),
                device=device,
            )
        )

    alpha_rows: list[dict[str, Any]] = []
    for value in ALPHA_VALUES:
        print(f"Running alpha sensitivity: {value}")
        alpha_rows.append(
            train_guided_gain(
                dataset=dataset,
                predictor=predictor,
                baseline_state=baseline_state,
                val_pairs=val_pairs,
                test_pairs=test_pairs,
                config=GuidedConfig(
                    name="Fine-tuned GAIN",
                    iterations=SWEEP_ITERATIONS,
                    alpha=value,
                    prediction_weight=DEFAULT_PREDICTION_WEIGHT,
                    model_dir=MODELS_DIR / f"alpha_{value:g}".replace(".", "p"),
                ),
                device=device,
            )
        )

    training_rows = [gain_metrics, guided_time_metrics]
    all_results = {
        "settings": {
            "seed": SEED,
            "miss_rate": MISS_RATE,
            "time_iterations": TIME_ITERATIONS,
            "sweep_iterations": SWEEP_ITERATIONS,
            "eval_pair_limit": EVAL_PAIR_LIMIT,
            "batch_size": BATCH_SIZE,
            "prediction_batch_size": PREDICTION_BATCH_SIZE,
            "default_alpha": DEFAULT_ALPHA,
            "default_prediction_weight": DEFAULT_PREDICTION_WEIGHT,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "training_time": training_rows,
        "lambda_sensitivity": lambda_rows,
        "alpha_sensitivity": alpha_rows,
    }
    formula_results = {
        "settings": {
            "seed": SEED,
            "miss_rate": MISS_RATE,
            "time_iterations": TIME_ITERATIONS,
            "sweep_iterations": SWEEP_ITERATIONS,
            "eval_pair_limit": EVAL_PAIR_LIMIT,
            "batch_size": BATCH_SIZE,
            "prediction_batch_size": PREDICTION_BATCH_SIZE,
            "default_lambda_2": DEFAULT_ALPHA,
            "default_lambda_3": DEFAULT_PREDICTION_WEIGHT,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "training_time": rows_with_formula_notation(training_rows),
        "lambda_3_sensitivity": rows_with_formula_notation(lambda_rows),
        "lambda_2_sensitivity": rows_with_formula_notation(alpha_rows),
    }

    write_csv(RESULTS_DIR / "training_time_summary.csv", rows_with_formula_notation(training_rows))
    write_csv(RESULTS_DIR / "lambda_3_sensitivity.csv", rows_with_formula_notation(lambda_rows))
    write_csv(RESULTS_DIR / "lambda_2_sensitivity.csv", rows_with_formula_notation(alpha_rows))
    with (RESULTS_DIR / "all_results_formula_notation.json").open("w", encoding="utf-8") as f:
        json.dump(formula_results, f, ensure_ascii=False, indent=2)

    plot_training_time(training_rows)
    plot_lambda(lambda_rows)
    plot_alpha(alpha_rows)
    write_result_docx(all_results)

    print("Experiment completed.")
    print(f"Plan: {EXP_DIR / 'experiment_plan.docx'}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Summary: {EXP_DIR / 'experiment_results_summary.docx'}")


if __name__ == "__main__":
    main()
