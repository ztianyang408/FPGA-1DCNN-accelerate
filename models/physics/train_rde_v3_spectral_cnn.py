"""Train a leakage-free single-PD spectral CNN on the compact RDE v3 subset."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SPLITS = ("train", "val", "test_id", "test_geometry", "test_hard")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spectral_features(root: Path, split: str, low_hz: float, high_hz: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    directory = root / split
    manifest = pd.read_csv(directory / "manifest.csv")
    cache: dict[str, np.ndarray] = {}
    frequency = np.fft.rfftfreq(20_000, d=1.0 / 20_000.0)
    mask = (frequency >= low_hz) & (frequency <= high_hz)
    band_frequency = frequency[mask]
    shapes: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    window = np.hanning(20_000).astype(np.float32)
    for row in manifest.itertuples(index=False):
        if row.shard_file not in cache:
            cache[row.shard_file] = np.load(directory / row.shard_file, mmap_mode="r")
        # The stored source is unit RMS by design. Restore the measured AC
        # voltage before spectral extraction, then keep RMS as an explicit input.
        voltage = cache[row.shard_file][int(row.row_in_shard)].astype(np.float32) * float(row.pre_normalization_rms_v)
        spectrum = np.abs(np.fft.rfft(voltage * window)) ** 2
        band = spectrum[mask] + 1e-18
        log_band = np.log(band)
        shapes.append(((log_band - log_band.mean()) / (log_band.std() + 1e-7)).astype(np.float32))
        power = float(band.sum())
        distribution = band / power
        centroid = float(np.sum(distribution * band_frequency))
        bandwidth = float(np.sqrt(np.sum(distribution * (band_frequency - centroid) ** 2)))
        peak = int(np.argmax(band))
        scalars.append(
            np.array(
                [
                    np.log(float(row.pre_normalization_rms_v) + 1e-18),
                    np.log(power),
                    centroid,
                    bandwidth,
                    band_frequency[peak],
                    np.log(float(band[peak] / (np.median(band) + 1e-18))),
                    -float(np.sum(distribution * np.log(distribution))) / np.log(len(distribution)),
                ],
                dtype=np.float32,
            )
        )
    return np.stack(shapes), np.stack(scalars), manifest


class SpectralCNN(nn.Module):
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
            nn.Linear(64 * 16 + scalar_count, 128), nn.GELU(), nn.Dropout(0.22), nn.Linear(128, 1)
        )

    def forward(self, shape: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        encoded = self.convolution(shape[:, None]).flatten(1)
        return self.head(torch.cat([encoded, scalar], dim=1)).squeeze(1)


def evaluate(model: nn.Module, shape: np.ndarray, scalar: np.ndarray, rpm_mean: float, rpm_std: float, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(shape), 256):
            output = model(torch.from_numpy(shape[start:start + 256]).to(device), torch.from_numpy(scalar[start:start + 256]).to(device))
            predictions.append(output.cpu().numpy())
    return (np.concatenate(predictions) * rpm_std + rpm_mean).astype(np.float32)


def metrics(rpm: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - rpm
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "mape_pct": float((np.abs(error) / rpm).mean() * 100.0),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
    }


def main() -> None:
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=parent / "data" / "rde_v3_stratified_18000")
    parser.add_argument("--output-dir", type=Path, default=parent / "results" / "rde_v3_spectral_cnn")
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

    train_shape, train_scalar, train_manifest = partitions["train_core"]
    scalar_mean = train_scalar.mean(axis=0)
    scalar_std = np.maximum(train_scalar.std(axis=0), 1e-6)
    normalized = {
        split: (shape, ((scalar - scalar_mean) / scalar_std).astype(np.float32), frame)
        for split, (shape, scalar, frame) in partitions.items()
    }
    train_rpm = train_manifest["rpm"].to_numpy(dtype=np.float32)
    rpm_mean, rpm_std = float(train_rpm.mean()), float(train_rpm.std())
    model = SpectralCNN(train_scalar.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_shape),
            torch.from_numpy(normalized["train_core"][1]),
            torch.from_numpy((train_rpm - rpm_mean) / rpm_std),
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
        for shape, scalar, rpm in loader:
            shape = shape.to(device)
            scalar = scalar.to(device)
            # Mild feature noise makes peak/amplitude inference less sensitive
            # to the finite simulation seeds without moving the frequency axis.
            shape = shape + 0.015 * torch.randn_like(shape)
            prediction = model(shape, scalar)
            loss = F.smooth_l1_loss(prediction, rpm.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        scheduler.step()
        val_shape, val_scalar, val_frame = normalized["val_core"]
        val_prediction = evaluate(model, val_shape, val_scalar, rpm_mean, rpm_std, device)
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
    torch.save(
        {"model_state": model.state_dict(), "rpm_mean": rpm_mean, "rpm_std": rpm_std, "scalar_mean": scalar_mean, "scalar_std": scalar_std},
        args.output_dir / "spectral_cnn_best.pt",
    )
    report: dict[str, object] = {
        "protocol": "Single-PD waveform only; voltage is recovered using its measured RMS. No geometry, theoretical frequency, f_mod_hz, recovered_rpm, or QC labels are model inputs.",
        "speed_protocol": {
            "train_core_rpm": [args.train_rpm_min, args.train_rpm_max],
            "low_speed_ood_rpm": [400.0, args.train_rpm_min],
            "high_speed_ood_rpm": [args.train_rpm_max, 3000.0],
            "checkpoint_selection": "val_core only",
        },
        "feature_band_hz": [20, 1600],
        "results": {},
    }
    outputs = []
    for split, (shape, scalar, frame) in normalized.items():
        prediction = evaluate(model, shape, scalar, rpm_mean, rpm_std, device)
        rpm = frame["rpm"].to_numpy(dtype=np.float32)
        report["results"][split] = {"samples": int(len(rpm)), **metrics(rpm, prediction)}
        saved = frame[["rpm", "severity", "gamma_deg", "phi_deg", "d_mm", "record_index"]].copy()
        saved["split"] = split
        saved["prediction_rpm"] = prediction
        saved["absolute_error_rpm"] = np.abs(prediction - rpm)
        outputs.append(saved)
    pd.concat(outputs, ignore_index=True).to_csv(args.output_dir / "per_sample_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
