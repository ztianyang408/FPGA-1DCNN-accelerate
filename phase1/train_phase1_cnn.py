from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "processed"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "phase1" / "results"
SPLITS = ("train", "val", "test_id", "test_geometry", "test_hard")


@dataclass
class SplitMetrics:
    split: str
    mae: float
    rmse: float
    mape: float
    r2: float


def load_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    split_dir = data_root / split
    x = np.load(split_dir / "features_fft.npy").astype(np.float32)
    y = np.load(split_dir / "targets_rpm.npy").astype(np.float32)
    meta = pd.read_csv(split_dir / "meta.csv")
    return x, y, meta


def evaluate_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> SplitMetrics:
    error = y_pred - y_true
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float((np.abs(error) / denom).mean() * 100.0)
    total = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - np.sum(error**2) / max(total, 1e-12))
    return SplitMetrics(split=name, mae=mae, rmse=rmse, mape=mape, r2=r2)


class LightweightRPMCNN(nn.Module):
    def __init__(self, input_len: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.head(x).squeeze(-1)


def predict_batches(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = (x[start : start + batch_size] - mean) / std
            batch_t = torch.from_numpy(batch).to(device)
            outputs.append(model(batch_t).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight 1D CNN on FFT features")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root / f"{timestamp}_cnn"
    run_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_meta = load_split(args.data_root, "train")
    x_val, y_val, val_meta = load_split(args.data_root, "val")
    x_test_id, y_test_id, test_id_meta = load_split(args.data_root, "test_id")
    x_test_geo, y_test_geo, test_geo_meta = load_split(args.data_root, "test_geometry")
    x_test_hard, y_test_hard, test_hard_meta = load_split(args.data_root, "test_hard")

    train_mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    train_std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(max(y_train.std(), 1e-6))

    x_train_n = (x_train - train_mean) / train_std
    x_val_n = (x_val - train_mean) / train_std
    x_test_id_n = (x_test_id - train_mean) / train_std
    x_test_geo_n = (x_test_geo - train_mean) / train_std
    x_test_hard_n = (x_test_hard - train_mean) / train_std

    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    train_ds = TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightRPMCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0

    print(f"Loaded data. train={x_train.shape}, val={x_val.shape}, test_id={x_test_id.shape}", flush=True)
    print(f"Training CNN on {device}...", flush=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_pred_n = predict_batches(model, x_val, train_mean, train_std, device)
        val_pred = val_pred_n * y_std + y_mean
        val_metrics = evaluate_split("val", y_val, val_pred)
        print(
            f"Epoch {epoch + 1:02d}: train_loss={np.mean(losses):.4f}, val_MAE={val_metrics.mae:.2f}, val_RMSE={val_metrics.rmse:.2f}, val_MAPE={val_metrics.mape:.2f}%",
            flush=True,
        )
        if val_metrics.mae < best_val:
            best_val = val_metrics.mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "model_state": model.state_dict(),
            "train_mean": train_mean,
            "train_std": train_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "input_len": int(x_train.shape[1]),
        },
        run_dir / "cnn_model.pt",
    )

    splits = [
        ("train", y_train, x_train, train_meta),
        ("val", y_val, x_val, val_meta),
        ("test_id", y_test_id, x_test_id, test_id_meta),
        ("test_geometry", y_test_geo, x_test_geo, test_geo_meta),
        ("test_hard", y_test_hard, x_test_hard, test_hard_meta),
    ]

    metrics = []
    predictions = []
    for split_name, y_true, x_split, meta in splits:
        print(f"Evaluating {split_name}...", flush=True)
        y_pred = predict_batches(model, x_split, train_mean, train_std, device) * y_std + y_mean
        split_metrics = evaluate_split(split_name, y_true, y_pred)
        metrics.append(asdict(split_metrics))
        pd.DataFrame(
            {
                "rpm_true": y_true,
                "rpm_pred": y_pred,
                "error": y_pred - y_true,
            }
        ).join(meta.reset_index(drop=True)).to_csv(run_dir / f"predictions_{split_name}.csv", index=False)
        predictions.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "rpm_true": y_true,
                    "rpm_pred": y_pred,
                    "error": y_pred - y_true,
                }
            )
        )
        print(
            f"Done {split_name}: MAE={split_metrics.mae:.3f}, RMSE={split_metrics.rmse:.3f}, MAPE={split_metrics.mape:.2f}%",
            flush=True,
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(run_dir / "all_predictions.csv", index=False)
    summary = {
        "model": "lightweight_1d_cnn",
        "data_root": str(args.data_root),
        "results_root": str(args.results_root),
        "run_dir": str(run_dir),
        "input_shape": [int(x_train.shape[1])],
        "normalization": {
            "train_mean": "per-feature train mean",
            "train_std": "per-feature train std",
            "target_mean": y_mean,
            "target_std": y_std,
        },
        "metrics": metrics,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved run to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
