# coding=utf-8
#
# Apache License 2.0
#

"""GAIN training script (PyTorch version)."""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from data_loader import data_loader
from gain import Discriminator, Generator, xavier_init_
from utils import (
    binary_sampler,
    normalization,
    renormalization,
    rmse_loss,
    rounding,
    sample_batch_index,
    uniform_sampler,
)


def gain_train(data_x, gain_parameters, device="cpu"):
    """Train a GAIN model and impute missing values."""
    data_m = 1 - np.isnan(data_x)

    batch_size = gain_parameters["batch_size"]
    hint_rate = gain_parameters["hint_rate"]
    alpha = gain_parameters["alpha"]
    iterations = gain_parameters["iterations"]

    no, dim = data_x.shape
    h_dim = int(dim)

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    norm_data_x_tensor = torch.FloatTensor(norm_data_x).to(device)
    data_m_tensor = torch.FloatTensor(data_m).to(device)

    generator = Generator(dim, h_dim).to(device)
    discriminator = Discriminator(dim, h_dim).to(device)
    generator.apply(xavier_init_)
    discriminator.apply(xavier_init_)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    print(f"Start training GAIN on device: {device}")
    print(f"Data shape: {data_x.shape}, hidden dimension: {h_dim}")

    for _ in tqdm(range(iterations)):
        batch_idx = sample_batch_index(no, batch_size)
        x_mb = norm_data_x_tensor[batch_idx, :]
        m_mb = data_m_tensor[batch_idx, :]

        z_mb = torch.FloatTensor(uniform_sampler(0, 0.01, batch_size, dim)).to(device)
        h_mb_temp = torch.FloatTensor(binary_sampler(hint_rate, batch_size, dim)).to(device)
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
        mse_loss = torch.mean((m_mb * x_mb - m_mb * g_sample) ** 2) / (
            torch.mean(m_mb) + 1e-8
        )
        g_loss = g_loss_adv + alpha * mse_loss
        g_loss.backward()
        g_optimizer.step()

    print("Training completed.")

    z_mb = torch.FloatTensor(uniform_sampler(0, 0.01, no, dim)).to(device)
    m_mb = data_m_tensor
    x_mb = norm_data_x_tensor
    x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

    with torch.no_grad():
        imputed_data = generator(x_mb, m_mb).cpu().numpy()

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    imputed_data = renormalization(imputed_data, norm_parameters)
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data, generator, norm_parameters


def main(args):
    """Load data, train GAIN, evaluate RMSE, and save artifacts."""
    gain_parameters = {
        "batch_size": args.batch_size,
        "hint_rate": args.hint_rate,
        "alpha": args.alpha,
        "iterations": args.iterations,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ori_data_x, miss_data_x, data_m = data_loader(args.data_name, args.miss_rate)
    print(f"Loaded dataset: {args.data_name}, shape: {ori_data_x.shape}")
    print(f"Auto-detected feature count: {ori_data_x.shape[1]}")
    print(f"Missing rate: {args.miss_rate}")

    imputed_data_x, generator, norm_parameters = gain_train(
        miss_data_x, gain_parameters, device
    )

    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    print()
    print(f"RMSE: {np.round(rmse, 4)}")

    os.makedirs(args.model_dir, exist_ok=True)
    generator_path = os.path.join(args.model_dir, "generator.pth")
    torch.save(generator.state_dict(), generator_path)
    print(f"Saved generator model to: {generator_path}")

    norm_params_path = os.path.join(args.model_dir, "norm_parameters.npy")
    np.save(norm_params_path, norm_parameters)
    print(f"Saved normalization parameters to: {norm_params_path}")

    return imputed_data_x, rmse, generator, norm_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        choices=[
            "letter",
            "spam",
            "LD2011_2014",
            "LD2011_2014_train",
            "electricity_data_2013_processed",
        ],
        default="electricity_data_2013_processed",
        type=str,
        help="数据集名称",
    )
    parser.add_argument("--miss_rate", default=0.2, type=float, help="缺失率")
    parser.add_argument("--batch_size", default=64, type=int, help="批量大小")
    parser.add_argument("--hint_rate", default=0.9, type=float, help="提示率")
    parser.add_argument("--alpha", default=100, type=float, help="超参数 alpha")
    parser.add_argument("--iterations", default=10000, type=int, help="训练迭代次数")
    parser.add_argument("--model_dir", default="saved_models", type=str, help="模型保存目录")

    args = parser.parse_args()
    imputed_data, rmse, generator, norm_parameters = main(args)
