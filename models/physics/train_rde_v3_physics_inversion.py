"""Physics-constrained single-PD inverse model for the RDE v3 subset.

The network has no RPM head and no independent four-frequency heads. It emits
only an effective modulation frequency and geometry. The paper's four-point
formula deterministically produces both the local audit frequencies and RPM.
Geometry/frequency labels are used only as synthetic calibration supervision.
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

from train_rde_v3_spectral_cnn import SPLITS, evaluate, set_seed, spectral_features


FREQUENCY_COLUMNS = ("f_theta0_hz", "f_theta90_hz", "f_theta180_hz", "f_theta270_hz")
RING_RADIUS_MM = 5.0


class PhysicsSpectralCNN(nn.Module):
    """PSD encoder whose outputs are physical latent variables only."""

    def __init__(self, scalar_count: int) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=9, stride=2, padding=4), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 16 + scalar_count, 128), nn.GELU(), nn.Dropout(0.22), nn.Linear(128, 5)
        )

    def forward(self, shape: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.convolution(shape[:, None]).flatten(1), scalar], dim=1))


def forward_physics(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implement paper_four_points_hz with differentiable constrained latents."""
    f_mod_hz = 500.0 * F.softplus(raw[:, 0])
    gamma_deg = 75.0 * torch.sigmoid(raw[:, 1])
    d_mm = 12.0 * torch.sigmoid(raw[:, 2])
    direction = F.normalize(raw[:, 3:5], dim=1, eps=1e-6)
    sin_phi, cos_phi = direction[:, 0], direction[:, 1]
    cos_gamma = torch.cos(torch.deg2rad(gamma_deg)).clamp(min=1e-3)
    q = d_mm / RING_RADIUS_MM
    f0 = f_mod_hz * cos_gamma * (1.0 - q * sin_phi)
    f90 = f_mod_hz * (1.0 / cos_gamma + q * cos_phi)
    f180 = f_mod_hz * cos_gamma * (1.0 + q * sin_phi)
    f270 = f_mod_hz * (1.0 / cos_gamma - q * cos_phi)
    local_frequency = torch.stack([f0, f90, f180, f270], dim=1)
    geometry = torch.stack([gamma_deg, d_mm, sin_phi, cos_phi], dim=1)
    return 3.0 * f_mod_hz, local_frequency, geometry


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "mape_pct": float((np.abs(error) / actual).mean() * 100.0),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
    }


def predict(model: nn.Module, shape: np.ndarray, scalar: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rpms, frequencies, geometries = [], [], []
    with torch.no_grad():
        for start in range(0, len(shape), 256):
            raw = model(torch.from_numpy(shape[start:start + 256]).to(device), torch.from_numpy(scalar[start:start + 256]).to(device))
            rpm, frequency, geometry = forward_physics(raw)
            rpms.append(rpm.cpu().numpy())
            frequencies.append(frequency.cpu().numpy())
            geometries.append(geometry.cpu().numpy())
    return np.concatenate(rpms), np.concatenate(frequencies), np.concatenate(geometries)


def main() -> None:
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=parent / "data" / "rde_v3_stratified_18000")
    parser.add_argument("--output-dir", type=Path, default=parent / "results" / "rde_v3_physics_inversion")
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
    train_rpm = train_frame["rpm"].to_numpy(dtype=np.float32)
    train_frequency = train_frame.loc[:, FREQUENCY_COLUMNS].to_numpy(dtype=np.float32)
    phi = np.deg2rad(train_frame["phi_deg"].to_numpy(dtype=np.float32))
    train_geometry = np.column_stack([train_frame["gamma_deg"], train_frame["d_mm"], np.sin(phi), np.cos(phi)]).astype(np.float32)
    frequency_mean = train_frequency.mean(axis=0)
    frequency_std = np.maximum(train_frequency.std(axis=0), 1e-5)
    geometry_scale = np.array([75.0, 12.0, 1.0, 1.0], dtype=np.float32)
    model = PhysicsSpectralCNN(train_scalar.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_shape), torch.from_numpy(datasets["train_core"][1]), torch.from_numpy(train_rpm),
            torch.from_numpy((train_frequency - frequency_mean) / frequency_std),
            torch.from_numpy(train_geometry / geometry_scale),
        ),
        batch_size=96,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae, patience = float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        for shape, scalar, rpm_target, frequency_target, geometry_target in loader:
            shape = shape.to(device) + 0.015 * torch.randn_like(shape.to(device))
            raw_output = model(shape, scalar.to(device))
            rpm_prediction, frequency_prediction, geometry_prediction = forward_physics(raw_output)
            rpm_loss = F.smooth_l1_loss(rpm_prediction / 1000.0, rpm_target.to(device) / 1000.0)
            frequency_loss = F.smooth_l1_loss(
                (frequency_prediction - torch.from_numpy(frequency_mean).to(device)) / torch.from_numpy(frequency_std).to(device),
                frequency_target.to(device),
            )
            geometry_loss = F.smooth_l1_loss(geometry_prediction / torch.from_numpy(geometry_scale).to(device), geometry_target.to(device))
            loss = rpm_loss + 0.45 * frequency_loss + 0.18 * geometry_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        scheduler.step()
        val_shape, val_scalar, val_frame = datasets["val_core"]
        val_prediction, _, _ = predict(model, val_shape, val_scalar, device)
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
    torch.save({"model_state": model.state_dict(), "scalar_mean": scalar_mean, "scalar_std": scalar_std}, args.output_dir / "physics_inversion_best.pt")
    report: dict[str, object] = {
        "protocol": "Waveform-only input. CNN outputs f_mod, gamma, d, sin(phi), cos(phi); paper four-point equations then produce the four audit frequencies and RPM=3*f_mod. Theory/geometry labels supervise training only.",
        "speed_protocol": {
            "train_core_rpm": [args.train_rpm_min, args.train_rpm_max],
            "low_speed_ood_rpm": [400.0, args.train_rpm_min],
            "high_speed_ood_rpm": [args.train_rpm_max, 3000.0],
            "checkpoint_selection": "val_core only",
        },
        "results": {},
    }
    saved = []
    for split, (shape, scalar, frame) in datasets.items():
        prediction, frequency_prediction, geometry_prediction = predict(model, shape, scalar, device)
        rpm = frame["rpm"].to_numpy(dtype=np.float32)
        frequency = frame.loc[:, FREQUENCY_COLUMNS].to_numpy(dtype=np.float32)
        true_phi = np.deg2rad(frame["phi_deg"].to_numpy(dtype=np.float32))
        true_geometry = np.column_stack([frame["gamma_deg"], frame["d_mm"], np.sin(true_phi), np.cos(true_phi)])
        report["results"][split] = {
            "samples": int(len(rpm)),
            **metrics(rpm, prediction),
            "frequency_mae_hz": float(np.abs(frequency_prediction - frequency).mean()),
            "gamma_mae_deg": float(np.abs(geometry_prediction[:, 0] - true_geometry[:, 0]).mean()),
            "d_mae_mm": float(np.abs(geometry_prediction[:, 1] - true_geometry[:, 1]).mean()),
        }
        rows = frame[["rpm", "severity", "gamma_deg", "phi_deg", "d_mm", "record_index"]].copy()
        rows["split"] = split
        rows["prediction_rpm"] = prediction
        rows["absolute_error_rpm"] = np.abs(prediction - rpm)
        rows["predicted_f_mod_hz"] = prediction / 3.0
        saved.append(rows)
    pd.concat(saved, ignore_index=True).to_csv(args.output_dir / "per_sample_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
