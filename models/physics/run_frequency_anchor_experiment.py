"""Frequency-anchored RPM experiment on the published dual-spot split.

The network predicts six PSD-derived beat-frequency distribution diagnostics.
RPM has no independent output head: a train-fitted linear calibration reads it
from those diagnostics. This makes the learned intermediate quantity explicit
and tests whether it generalizes better than direct RPM regression.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from run_dualspot_formal_experiment import (
    FeatureConfig,
    SharedEncoder,
    load_split,
    normalize_features,
    regression_metrics,
    set_seed,
)


class FrequencyAnchorRegressor(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.encoder = SharedEncoder(inputs, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def fit_frequency_calibration(frequency: np.ndarray, rpm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = frequency.mean(axis=0).astype(np.float32)
    std = np.maximum(frequency.std(axis=0), 1e-5).astype(np.float32)
    x = (frequency - mean) / std
    # Mild ridge regularization prevents highly correlated quantiles from
    # yielding a numerically unstable RPM readout.
    coefficient = np.linalg.solve(x.T @ x + 10.0 * np.eye(x.shape[1]), x.T @ (rpm - rpm.mean())).astype(np.float32)
    return mean, std, coefficient, float(rpm.mean())


def predict_rpm(model: nn.Module, x: np.ndarray, coefficient: np.ndarray, rpm_intercept: float, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        frequency_normalized = model(torch.from_numpy(x).to(device))
        rpm = frequency_normalized @ torch.from_numpy(coefficient).to(device) + rpm_intercept
    return rpm.cpu().numpy().astype(np.float32)


def train(
    train_x: np.ndarray,
    train_rpm: np.ndarray,
    train_frequency: np.ndarray,
    val_x: np.ndarray,
    val_rpm: np.ndarray,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    epochs: int,
    device: torch.device,
) -> FrequencyAnchorRegressor:
    frequency_mean, frequency_std, coefficient, rpm_intercept = calibration
    normalized_frequency = (train_frequency - frequency_mean) / frequency_std
    model = FrequencyAnchorRegressor(train_x.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train_rpm), torch.from_numpy(normalized_frequency)
        ),
        batch_size=64,
        shuffle=True,
    )
    coefficient_tensor = torch.from_numpy(coefficient).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae, patience = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for features, rpm, frequency in loader:
            predicted_frequency = model(features.to(device))
            predicted_rpm = predicted_frequency @ coefficient_tensor + rpm_intercept
            frequency_loss = F.smooth_l1_loss(predicted_frequency, frequency.to(device))
            rpm_loss = F.smooth_l1_loss(predicted_rpm / 400.0, rpm.to(device) / 400.0)
            optimizer.zero_grad()
            (0.55 * frequency_loss + rpm_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        prediction = predict_rpm(model, val_x, coefficient, rpm_intercept, device)
        mae = regression_metrics(val_rpm, prediction)["mae_rpm"]
        if mae < best_mae:
            best_mae, patience = mae, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 35:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parent = Path(__file__).resolve().parent
    parser.add_argument("--data-root", default=str(parent / "data" / "dual_spot_theta0_pi_formal_extrap_csv_v1"))
    parser.add_argument("--output-dir", default=str(parent / "results" / "frequency_anchor_experiment"))
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    manifest = pd.read_csv(root / "manifest.csv")
    config = FeatureConfig()
    splits = ("train", "val", "test_in", "test_edge_low", "test_edge_high", "test_ood_low", "test_ood_high")
    data = {split: load_split(manifest, split, root, config) for split in splits}
    raw_train = data["train"][0]
    normalized: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for split, (x, rpm, geometry, frequency, frame) in data.items():
        x_normalized, _, _ = normalize_features(raw_train, x)
        normalized[split] = (x_normalized, rpm, geometry, frequency, frame)
    train_x, train_rpm, _, train_frequency, _ = normalized["train"]
    val_x, val_rpm, _, _, _ = normalized["val"]
    calibration = fit_frequency_calibration(train_frequency, train_rpm)
    model = train(train_x, train_rpm, train_frequency, val_x, val_rpm, calibration, args.epochs, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "frequency_calibration": calibration}, output / "frequency_anchor_best.pt")
    report: dict[str, object] = {
        "protocol": "RPM is read only from six network-predicted frequency diagnostics using a train-split ridge calibration.",
        "results": {},
    }
    rows: list[pd.DataFrame] = []
    for split, (x, rpm, _, frequency, frame) in normalized.items():
        prediction = predict_rpm(model, x, calibration[2], calibration[3], device)
        oracle = ((frequency - calibration[0]) / calibration[1]) @ calibration[2] + calibration[3]
        report["results"][split] = {
            "frequency_anchor": regression_metrics(rpm, prediction),
            "oracle_frequency_calibration": regression_metrics(rpm, oracle),
        }
        saved = frame.copy()
        saved["frequency_anchor_prediction_rpm"] = prediction
        saved["frequency_anchor_absolute_error_rpm"] = np.abs(prediction - rpm)
        rows.append(saved)
    pd.concat(rows, ignore_index=True).to_csv(output / "per_sample_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
