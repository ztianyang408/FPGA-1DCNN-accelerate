"""FFT-reconstruction physics model for single-PD RDE speed extrapolation.

The target rotates at f_rot.  A fixed rough-surface response contains angular
orders m, so its rotation-induced spectral components must be centred at
m*f_rot.  The CNN predicts f_rot plus nuisance harmonic powers/linewidths;
the differentiable comb layer reconstructs the measured log-PSD shape.
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

from train_rde_v3_spectral_cnn import SPLITS, set_seed, spectral_features


MAX_ORDER = 64
PSD_STRIDE = 4


class FFTCombCNN(nn.Module):
    def __init__(self, scalar_count: int) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=9, stride=2, padding=4), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        # f_rot, shared linewidth, background level/slope, then 64 positive
        # effective angular-order powers.
        self.head = nn.Sequential(
            nn.Linear(64 * 16 + scalar_count, 128), nn.GELU(), nn.Dropout(0.22), nn.Linear(128, MAX_ORDER + 4)
        )

    def forward(self, shape: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.convolution(shape[:, None]).flatten(1), scalar], dim=1))


def comb_forward(raw: torch.Tensor, frequency_hz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a normalized log-PSD from harmonics at m*f_rot."""
    f_rot = 30.0 * F.softplus(raw[:, 0])
    linewidth = 1.0 + 8.0 * F.softplus(raw[:, 1])
    background_level = F.softplus(raw[:, 2]) + 1e-5
    background_slope = raw[:, 3].clamp(-4.0, 4.0)
    amplitude = F.softplus(raw[:, 4:])
    orders = torch.arange(1, MAX_ORDER + 1, dtype=raw.dtype, device=raw.device)[None, :, None]
    frequency = frequency_hz[None, None, :]
    centres = f_rot[:, None, None] * orders
    # Higher angular orders broaden somewhat more under speed ripple/wander.
    widths = linewidth[:, None, None] * (1.0 + 0.012 * orders)
    kernels = torch.exp(-0.5 * ((frequency - centres) / widths).square())
    harmonic_power = torch.sum(amplitude[:, :, None] * kernels, dim=1)
    normalized_frequency = (frequency_hz - frequency_hz.mean()) / (frequency_hz.max() - frequency_hz.min())
    background = background_level[:, None] * torch.exp(background_slope[:, None] * normalized_frequency[None])
    log_psd = torch.log(harmonic_power + background + 1e-8)
    psd_shape = (log_psd - log_psd.mean(dim=1, keepdim=True)) / (log_psd.std(dim=1, keepdim=True) + 1e-6)
    return 60.0 * f_rot, psd_shape, amplitude


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "mape_pct": float((np.abs(error) / actual).mean() * 100.0),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
    }


def predict(model: nn.Module, shape: np.ndarray, scalar: np.ndarray, frequency_hz: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rpm_parts, shape_parts = [], []
    with torch.no_grad():
        for start in range(0, len(shape), 256):
            raw = model(torch.from_numpy(shape[start:start + 256]).to(device), torch.from_numpy(scalar[start:start + 256]).to(device))
            rpm, reconstructed, _ = comb_forward(raw, frequency_hz)
            rpm_parts.append(rpm.cpu().numpy())
            shape_parts.append(reconstructed.cpu().numpy())
    return np.concatenate(rpm_parts), np.concatenate(shape_parts)


def main() -> None:
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=parent / "data" / "rde_v3_stratified_18000")
    parser.add_argument("--output-dir", type=Path, default=parent / "results" / "rde_v3_fft_comb_physics")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--train-rpm-min", type=float, default=700.0)
    parser.add_argument("--train-rpm-max", type=float, default=2400.0)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = {split: spectral_features(args.data_root, split, 20.0, 1_600.0) for split in SPLITS}

    def partition(source: str, name: str, lower: float | None, upper: float | None) -> tuple[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
        shape, scalar, frame = raw[source]
        mask = np.ones(len(frame), dtype=bool)
        if lower is not None:
            mask &= frame["rpm"].to_numpy() >= lower
        if upper is not None:
            mask &= frame["rpm"].to_numpy() <= upper
        if not mask.any():
            raise ValueError(f"No samples in partition {name}")
        return name, (shape[mask], scalar[mask], frame.loc[mask].reset_index(drop=True))

    partitions = dict(
        [
            partition("train", "train_core", args.train_rpm_min, args.train_rpm_max),
            partition("val", "val_core", args.train_rpm_min, args.train_rpm_max),
        ]
    )
    for source in ("test_id", "test_geometry", "test_hard"):
        partitions.update(
            [
                partition(source, f"{source}_core", args.train_rpm_min, args.train_rpm_max),
                partition(source, f"{source}_ood_low", None, args.train_rpm_min - 1e-6),
                partition(source, f"{source}_ood_high", args.train_rpm_max + 1e-6, None),
            ]
        )
    train_shape, train_scalar, train_frame = partitions["train_core"]
    scalar_mean, scalar_std = train_scalar.mean(axis=0), np.maximum(train_scalar.std(axis=0), 1e-6)
    datasets = {
        split: (shape, ((scalar - scalar_mean) / scalar_std).astype(np.float32), frame)
        for split, (shape, scalar, frame) in partitions.items()
    }
    frequency_hz = torch.arange(20.0, 1600.0 + 1e-4, float(PSD_STRIDE), device=device)
    train_rpm = train_frame["rpm"].to_numpy(dtype=np.float32)
    train_psd_shape = train_shape[:, ::PSD_STRIDE]
    model = FFTCombCNN(train_scalar.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_shape), torch.from_numpy(datasets["train_core"][1]),
            torch.from_numpy(train_rpm), torch.from_numpy(train_psd_shape),
        ),
        batch_size=72,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae, patience = float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        for shape, scalar, rpm_target, psd_target in loader:
            shape = shape.to(device) + 0.012 * torch.randn_like(shape.to(device))
            raw_output = model(shape, scalar.to(device))
            rpm_prediction, reconstructed_psd, amplitude = comb_forward(raw_output, frequency_hz)
            rpm_loss = F.smooth_l1_loss(rpm_prediction / 1000.0, rpm_target.to(device) / 1000.0)
            psd_loss = F.smooth_l1_loss(reconstructed_psd, psd_target.to(device))
            sparsity_loss = amplitude.mean()
            loss = rpm_loss + 0.32 * psd_loss + 0.0005 * sparsity_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        scheduler.step()
        val_shape, val_scalar, val_frame = datasets["val_core"]
        val_prediction, _ = predict(model, val_shape, val_scalar, frequency_hz, device)
        val_mae = metrics(val_frame["rpm"].to_numpy(dtype=np.float32), val_prediction)["mae_rpm"]
        if val_mae < best_mae:
            best_mae, patience = val_mae, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 14:
                break
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: validation MAE {val_mae:.2f} RPM")
    assert best_state is not None
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "scalar_mean": scalar_mean, "scalar_std": scalar_std}, args.output_dir / "fft_comb_physics_best.pt")
    report: dict[str, object] = {
        "protocol": "Waveform-only input. CNN predicts f_rot and nuisance harmonic powers; a differentiable m*f_rot comb reconstructs the observed FFT shape, and RPM=60*f_rot. No theoretical frequency, geometry, or RPM-derived field is an input.",
        "speed_protocol": {
            "train_core_rpm": [args.train_rpm_min, args.train_rpm_max],
            "low_speed_ood_rpm": [400.0, args.train_rpm_min],
            "high_speed_ood_rpm": [args.train_rpm_max, 3000.0],
            "checkpoint_selection": "val_core only",
        },
        "comb_orders": [1, MAX_ORDER],
        "results": {},
    }
    saved = []
    for split, (shape, scalar, frame) in datasets.items():
        prediction, reconstructed = predict(model, shape, scalar, frequency_hz, device)
        rpm = frame["rpm"].to_numpy(dtype=np.float32)
        psd_mae = float(np.abs(reconstructed - shape[:, ::PSD_STRIDE]).mean())
        report["results"][split] = {"samples": int(len(rpm)), **metrics(rpm, prediction), "psd_shape_mae": psd_mae}
        rows = frame[["rpm", "severity", "gamma_deg", "phi_deg", "d_mm", "record_index"]].copy()
        rows["split"] = split
        rows["prediction_rpm"] = prediction
        rows["absolute_error_rpm"] = np.abs(prediction - rpm)
        rows["predicted_f_rot_hz"] = prediction / 60.0
        saved.append(rows)
    pd.concat(saved, ignore_index=True).to_csv(args.output_dir / "per_sample_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
