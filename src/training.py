import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from .data import prepare_sequences, standardize_sequences
from .models import JointCompressorTransformer, LSTM, TransformerPredictor


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, train_loader, device, epochs=50, lr=1e-3, weight_decay=0.0):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()


def get_predictions(model, test_loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).cpu().numpy()
            preds.append(pred)
            true.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(true, axis=0)


def bivariate_correlation(y_true, y_pred):
    numerator = np.sum(y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1])
    denominator = np.sqrt(
        np.sum(y_true[:, 0] ** 2 + y_true[:, 1] ** 2)
        * np.sum(y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2)
    )
    return numerator / denominator


def correlation_1d(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def summarize_results(results, output_dim):
    results = sorted(results, key=lambda r: r["test_year"])
    all_pred = np.concatenate([r["pred"] for r in results], axis=0)
    all_true = np.concatenate([r["true"] for r in results], axis=0)
    summary = {
        "avg_mse": float(np.mean([r["mse"] for r in results])),
        "avg_r2": float(np.mean([r["r2"] for r in results])),
    }
    if output_dim == 2:
        summary["bivariate_corr"] = float(bivariate_correlation(all_true, all_pred))
    else:
        summary["corr"] = correlation_1d(all_true, all_pred)
    return summary


def make_loaders(X_train, X_test, y_train, y_test, batch_size):
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def build_model(model_name, input_dim, output_dim, config):
    if model_name == "lstm":
        cfg = config["models"]["lstm"]
        return LSTM(
            latent_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            output_dim=output_dim,
            dropout_rate=cfg["dropout_rate"],
        )
    if model_name == "transformer":
        cfg = config["models"]["transformer"]
        return TransformerPredictor(
            latent_dim=input_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            output_dim=output_dim,
            dropout_rate=cfg["dropout_rate"],
        )
    if model_name == "joint":
        return JointCompressorTransformer(
            input_dim=input_dim,
            compressor_cfg=config["models"]["joint"]["compressor"],
            transformer_cfg=config["models"]["transformer"],
            output_dim=output_dim,
        )
    raise ValueError(f"Unknown model: {model_name}")


def run_one_fold(model_name, test_year, X_seq, y_seq, seq_years, config, device):
    train_mask = seq_years != test_year
    test_mask = seq_years == test_year
    X_train, X_test = standardize_sequences(X_seq[train_mask], X_seq[test_mask])
    y_train, y_test = y_seq[train_mask], y_seq[test_mask]

    train_cfg = config["training"][model_name]
    train_loader, test_loader = make_loaders(
        X_train, X_test, y_train, y_test, batch_size=train_cfg["batch_size"]
    )
    model = build_model(model_name, X_seq.shape[-1], y_seq.shape[-1], config).to(device)
    train_model(
        model,
        train_loader,
        device=device,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    pred, true = get_predictions(model, test_loader, device)
    return {
        "test_year": int(test_year),
        "pred": pred,
        "true": true,
        "mse": float(mean_squared_error(true, pred)),
        "r2": float(r2_score(true, pred, multioutput="uniform_average")),
    }


def run_model(model_name, latent_name, sequences, config, device):
    X_seq = sequences[latent_name]
    y_seq = sequences["target"]
    seq_years = sequences["years"]
    results = []
    for test_year in sorted(np.unique(seq_years).tolist()):
        results.append(run_one_fold(model_name, test_year, X_seq, y_seq, seq_years, config, device))
    return results


def run_experiment(config):
    set_seed(config.get("seed", 42))
    device = get_device()
    output_dim = len(config["target"]["columns"])

    prepared_sequences = prepare_sequences(config, config["sequence"]["seq_len"])

    all_results = {}
    summary_rows = []
    for model_name, sequences in [
        ("lstm", prepared_sequences),
        ("transformer", prepared_sequences),
        ("joint", prepared_sequences),
    ]:
        for latent_name in ["encoder", "decoder"]:
            key = f"{model_name}_{latent_name}"
            results = run_model(model_name, latent_name, sequences, config, device)
            all_results[key] = results
            summary_rows.append(
                {
                    "model": model_name,
                    "latent": latent_name,
                    **summarize_results(results, output_dim),
                }
            )

    return all_results, pd.DataFrame(summary_rows)
