import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import pickle
from typing import List, Tuple

# -------------------------- 参数设置（无修改） --------------------------
class Config:
    data_path = "electricity_data_2013.csv"  # 数据路径
    lookback = 24*30  # 30天×24小时
    horizon = 24*7    # 7天×24小时
    test_ratio = 0.2
    val_ratio = 0.1
    
    # 模型参数
    conv_kernel = 6
    conv_channels = 32
    lstm_hidden = 128
    ar_window_daily = 24
    ar_window_weekly = 168
    user_emb_dim = 8
    dropout_rate = 0.2
    
    # 训练参数
    batch_size = 128
    epochs = 100
    lr = 5e-4
    patience = 15
    huber_delta = 1.5
    weight_decay = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalers_save_path = "lstnet_scalers_optimized.pkl"
    model_save_path = "lstnet_best_model_optimized.pth"
    train_curve_path = "lstnet_training_curve_optimized.png"
    pred_example_path = "lstnet_prediction_examples_optimized.png"

config = Config()

# -------------------------- 数据集类（无修改） --------------------------
class ElectricityDataset(Dataset):
    def __init__(self, X, y, user_indices):
        self.X = X
        self.y = y
        self.user_indices = user_indices
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.X[idx]), 
                torch.FloatTensor(self.y[idx]), 
                self.user_indices[idx])

# -------------------------- 特征工程（无修改） --------------------------
def create_time_features(timestamps: np.ndarray) -> np.ndarray:
    features = []
    
    for ts_str in timestamps:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H")
        
        hour = ts.hour
        weekday = ts.weekday()
        month = ts.month
        is_weekend = 1 if weekday >= 5 else 0
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        weekday_sin = np.sin(2 * np.pi * weekday / 7)
        weekday_cos = np.cos(2 * np.pi * weekday / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        season = (month - 1) // 3 + 1
        season_sin = np.sin(2 * np.pi * season / 4)
        season_cos = np.cos(2 * np.pi * season / 4)
        
        features.append([
            hour_sin, hour_cos, 
            weekday_sin, weekday_cos,
            month_sin, month_cos,
            is_weekend, season_sin, season_cos
        ])
    return np.array(features)

def add_stat_features(series: np.ndarray, lookback: int) -> np.ndarray:
    n_timesteps = len(series)
    stat_feats = np.zeros((n_timesteps, 4))
    
    for i in range(n_timesteps):
        window_start = max(0, i - lookback + 1)
        window = series[window_start:i+1]
        
        mean_val = np.mean(window)
        std_val = np.std(window)
        max_val = np.max(window)
        peak_ratio = max_val / (mean_val + 1e-8)
        
        stat_feats[i] = [mean_val, std_val, max_val, peak_ratio]
    return stat_feats

# -------------------------- 数据预处理（无修改） --------------------------
def prepare_data() -> Tuple[DataLoader, DataLoader, DataLoader, List[MinMaxScaler], Tuple[np.ndarray, np.ndarray, np.ndarray], int]:
    df = pd.read_csv(config.data_path, header=None)
    timestamps = df.iloc[:, 0].values
    user_data = df.iloc[:, 1:].values
    n_users = user_data.shape[1]
    n_timesteps = user_data.shape[0]
    print(f"Data loaded: {n_timesteps} timesteps, {n_users} users (auto-detected)")

    time_features = create_time_features(timestamps)
    print(f"Time features generated: shape={time_features.shape}")

    scalers = []
    normalized_data = []
    for user_idx in range(n_users):
        scaler = MinMaxScaler(feature_range=(0, 1))
        user_series = user_data[:, user_idx].reshape(-1, 1)
        normalized_series = scaler.fit_transform(user_series).flatten()
        normalized_data.append(normalized_series)
        scalers.append(scaler)
    normalized_data = np.array(normalized_data).T
    print(f"User data normalized: shape={normalized_data.shape}")

    with open(config.scalers_save_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"User scalers saved to: {config.scalers_save_path}")

    all_stat_features = []
    for user_idx in range(n_users):
        user_series = normalized_data[:, user_idx]
        stat_feats = add_stat_features(user_series, config.lookback)
        all_stat_features.append(stat_feats)
    all_stat_features = np.array(all_stat_features).transpose(1, 0, 2)
    print(f"Statistical features generated: shape={all_stat_features.shape}")

    X_list, y_list, user_indices_list = [], [], []
    for user_idx in range(n_users):
        elec_series = normalized_data[:, user_idx]
        time_feats = time_features
        stat_feats = all_stat_features[:, user_idx]
        
        n_samples = n_timesteps - config.lookback - config.horizon + 1
        if n_samples <= 0:
            print(f"Warning: User {user_idx} has insufficient data to generate samples")
            continue
        
        for i in range(n_samples):
            x_elec = elec_series[i:i+config.lookback].reshape(-1, 1)
            x_time = time_feats[i:i+config.lookback]
            x_stat = stat_feats[i:i+config.lookback]
            x = np.concatenate([x_elec, x_time, x_stat], axis=1)
            
            y_ = elec_series[i+config.lookback : i+config.lookback+config.horizon].reshape(-1, 1)
            
            X_list.append(x)
            y_list.append(y_)
            user_indices_list.append(user_idx)

    X = np.array(X_list)
    y = np.array(y_list)
    user_indices = np.array(user_indices_list)
    print(f"Samples generated: total={len(X)}, X.shape={X.shape}, y.shape={y.shape}")

    user_sample_indices = {u: [] for u in range(n_users)}
    for idx, u in enumerate(user_indices):
        user_sample_indices[u].append(idx)
    
    train_indices, val_indices, test_indices = [], [], []
    for u in user_sample_indices:
        u_indices = np.array(user_sample_indices[u])
        n_u = len(u_indices)
        if n_u == 0:
            continue
        
        test_size = int(n_u * config.test_ratio)
        test_u = u_indices[-test_size:] if test_size > 0 else []
        train_val_u = u_indices[:-test_size] if test_size > 0 else u_indices
        
        val_size = int(len(train_val_u) * config.val_ratio)
        val_u = train_val_u[-val_size:] if val_size > 0 else []
        train_u = train_val_u[:-val_size] if val_size > 0 else train_val_u
        
        train_indices.extend(train_u)
        val_indices.extend(val_u)
        test_indices.extend(test_u)
    
    X_train, y_train, ui_train = X[train_indices], y[train_indices], user_indices[train_indices]
    X_val, y_val, ui_val = X[val_indices], y[val_indices], user_indices[val_indices]
    X_test, y_test, ui_test = X[test_indices], y[test_indices], user_indices[test_indices]
    print(f"Dataset split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    train_dataset = ElectricityDataset(X_train, y_train, ui_train)
    val_dataset = ElectricityDataset(X_val, y_val, ui_val)
    test_dataset = ElectricityDataset(X_test, y_test, ui_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scalers, (X_test, y_test, ui_test), n_users

# -------------------------- 模型定义（无修改） --------------------------
class LSTNetOptimized(nn.Module):
    def __init__(self, n_users: int):
        super(LSTNetOptimized, self).__init__()
        self.config = config
        
        self.user_emb = nn.Embedding(
            num_embeddings=n_users,
            embedding_dim=config.user_emb_dim,
            padding_idx=-1
        )
        
        input_dim = 14
        self.conv = nn.Conv1d(
            in_channels=input_dim + config.user_emb_dim,
            out_channels=config.conv_channels,
            kernel_size=config.conv_kernel,
            stride=1,
            padding=config.conv_kernel - 1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        
        self.lstm = nn.LSTM(
            input_size=config.conv_channels,
            hidden_size=config.lstm_hidden,
            batch_first=True,
            bidirectional=False
        )
        
        self.ar_fc_daily = nn.Linear(config.ar_window_daily, config.horizon)
        self.ar_fc_weekly = nn.Linear(config.ar_window_weekly, config.horizon)
        self.ar_fusion = nn.Linear(2, 1)
        
        self.fc = nn.Linear(config.lstm_hidden, config.horizon)
    
    def forward(self, x: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        user_feat = self.user_emb(user_idx)
        user_feat = user_feat.unsqueeze(1).repeat(1, self.config.lookback, 1)
        x = torch.cat([x, user_feat], dim=2)
        
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv[:, :, :-self.config.conv_kernel+1]
        x_conv = self.relu(x_conv)
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x_conv)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        main_out = self.fc(lstm_out)
        
        elec_feat = x[:, :, 0]
        ar_input_daily = elec_feat[:, -self.config.ar_window_daily:]
        ar_out_daily = self.ar_fc_daily(ar_input_daily)
        ar_input_weekly = elec_feat[:, -self.config.ar_window_weekly:]
        ar_out_weekly = self.ar_fc_weekly(ar_input_weekly)
        ar_out = torch.stack([ar_out_daily, ar_out_weekly], dim=-1)
        ar_out = self.ar_fusion(ar_out).squeeze(-1)
        
        total_out = main_out + ar_out
        return total_out.unsqueeze(-1)

# -------------------------- 损失函数（无修改） --------------------------
class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        return torch.mean(loss)

# -------------------------- 训练函数（无修改） --------------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[nn.Module, dict]:
    model.to(config.device)
    criterion = HuberLoss(delta=config.huber_delta)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4,
        verbose=True,
        min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, user_idx_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            user_idx_batch = user_idx_batch.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(X_batch, user_idx_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, user_idx_batch in val_loader:
                X_batch = X_batch.to(config.device)
                y_batch = y_batch.to(config.device)
                user_idx_batch = user_idx_batch.to(config.device)
                
                outputs = model(X_batch, user_idx_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1} | Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
    return model, history

# -------------------------- 评估函数（画图标签改为英文） --------------------------
def evaluate_model(model: nn.Module, test_loader: DataLoader, scalers: List[MinMaxScaler], test_data: Tuple[np.ndarray, np.ndarray, np.ndarray], history: dict):
    model.to(config.device)
    model.eval()
    X_test, y_test, ui_test = test_data
    
    all_preds = []
    all_trues = []
    all_users = []
    with torch.no_grad():
        for X_batch, y_batch, u_batch in test_loader:
            X_batch = X_batch.to(config.device)
            u_batch = u_batch.to(config.device)  # 可能被转移到GPU
            outputs = model(X_batch, u_batch)
            # 以下三行均需确保张量在CPU上再转NumPy
            all_preds.append(outputs.cpu().numpy())  # 已正确处理
            all_trues.append(y_batch.numpy())  # 如果y_batch在CPU则无需改，若在GPU也需加.cpu()
            all_users.extend(u_batch.cpu().numpy())  # 修正此处，添加.cpu()
    
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    users = np.array(all_users)
    
    mse_list, mae_list = [], []
    for i in range(len(preds)):
        user_idx = int(users[i])
        scaler = scalers[user_idx]
        pred_denorm = scaler.inverse_transform(preds[i])
        true_denorm = scaler.inverse_transform(trues[i])
        mse = np.mean((pred_denorm - true_denorm) ** 2)
        mae = np.mean(np.abs(pred_denorm - true_denorm))
        mse_list.append(mse)
        mae_list.append(mae)
    
    overall_mse = np.mean(mse_list)
    overall_mae = np.mean(mae_list)
    print(f"\nOptimized Model Test Performance | MSE: {overall_mse:.6f} | MAE: {overall_mae:.6f}")
    
    # 预测示例图（标签改为英文）
    n_vis = 3
    if len(preds) < n_vis:
        n_vis = len(preds)
    sample_indices = list(range(n_vis))
    
    plt.figure(figsize=(12, 18))
    for i, sample_idx in enumerate(sample_indices):
        user_idx = int(users[sample_idx])
        scaler = scalers[user_idx]
        
        pred_denorm = scaler.inverse_transform(preds[sample_idx]).flatten()
        true_denorm = scaler.inverse_transform(trues[sample_idx]).flatten()
        
        plt.subplot(3, 1, i+1)
        plt.plot(range(len(pred_denorm)), pred_denorm, label="Predicted Value", alpha=0.8, linewidth=2)
        plt.plot(range(len(true_denorm)), true_denorm, label="True Value", alpha=0.8, linewidth=2)
        plt.title(f"User {user_idx} Electricity Consumption Prediction (Sample {i+1})", fontsize=14)
        plt.xlabel("Predicted Hour", fontsize=12)
        plt.ylabel("Electricity Consumption", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.pred_example_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 训练曲线图（标签改为英文）
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label="Training Loss", linewidth=2)
    plt.plot(history['val_loss'], label="Validation Loss", linewidth=2)
    plt.title("Model Training Loss Curve", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Huber Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(config.train_curve_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return overall_mse, overall_mae

# -------------------------- 主函数（无修改） --------------------------
if __name__ == "__main__":
    print("="*50)
    print("Start Training & Evaluation of Optimized LSTNet ")
    print("="*50)
    
    print("\n1. Preprocessing data...")
    train_loader, val_loader, test_loader, scalers, test_data, n_users = prepare_data()
    
    print("\n2. Initializing optimized LSTNet model...")
    model = LSTNetOptimized(n_users=n_users)
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"User embedding config: {n_users} users × {config.user_emb_dim}-dimensional embedding")
    
    print("\n3. Starting model training...")
    model, history = train_model(model, train_loader, val_loader)
    
    print("\n4. Starting model evaluation...")
    evaluate_model(model, test_loader, scalers, test_data, history)
    
    print("\n" + "="*50)
    print("Optimized Model Training & Evaluation Completed!")
    print("="*50)
