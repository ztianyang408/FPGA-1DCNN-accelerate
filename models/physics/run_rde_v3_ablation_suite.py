"""Run the presentation-ready RDE v3 ablation study.

Models:
  A0: raw time waveform -> f_rot -> RPM
  A1: FFT log-PSD -> f_rot -> RPM
  A2: A1 + geometry multitask supervision
  A3: A2 + Equation (10) four-point frequency supervision
  A4: A3 + differentiable FFT-spectrum reconstruction consistency

All models use the same predefined samples, 700-2400 RPM training interval,
validation-only checkpoint selection, optimizer, seed, and comparable encoder
capacity.  A1-A4 use PSD only; no hand-crafted scalar summary is an input.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from train_rde_v3_geometry_comb_physics import (
    FREQUENCY_COLUMNS,
    GeometryCombCNN,
    comb_forward,
    physical_four_points,
)
from train_rde_v3_spectral_cnn import SPLITS, set_seed, spectral_features


PSD_STRIDE = 4
MODEL_NAMES = {
    "A0": "Time-domain CNN",
    "A1": "FFT CNN",
    "A2": "FFT + geometry multitask",
    "A3": "FFT + geometry + Equation (10)",
    "A4": "FFT + geometry + Equation (10) + spectrum consistency",
}


class SignalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 24, 15, stride=2, padding=7), nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 48, 9, stride=2, padding=4), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x[:, None]).flatten(1)


class PhysicalOutputCNN(nn.Module):
    """Common encoder/head for A0-A3; unused outputs are ignored by ablation."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SignalEncoder()
        self.head = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Dropout(0.20), nn.Linear(128, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class MMapWaveformDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, data_root: Path, source_split: str, frame: pd.DataFrame, training: bool) -> None:
        self.directory = data_root / source_split
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.frame.iloc[index]
        filename = str(row["shard_file"])
        if filename not in self.cache:
            self.cache[filename] = np.load(self.directory / filename, mmap_mode="r")
        waveform = np.array(self.cache[filename][int(row["row_in_shard"])], dtype=np.float32, copy=True)
        if self.training:
            waveform += np.random.normal(0.0, 0.008, size=waveform.shape).astype(np.float32)
        targets = targets_from_frame(self.frame.iloc[[index]])
        return (
            torch.from_numpy(waveform),
            torch.tensor(float(row["rpm"]), dtype=torch.float32),
            torch.from_numpy(targets[0][0]),
            torch.from_numpy(targets[1][0]),
        )


def targets_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frequency = frame.loc[:, FREQUENCY_COLUMNS].to_numpy(dtype=np.float32)
    phi = np.deg2rad(frame["phi_deg"].to_numpy(dtype=np.float32))
    geometry = np.column_stack(
        [frame["gamma_deg"].to_numpy() / 75.0, frame["d_mm"].to_numpy() / 12.0, np.sin(phi), np.cos(phi)]
    ).astype(np.float32)
    return frequency, geometry


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    residual = float(np.sum(error**2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "mape_pct": float((np.abs(error) / actual).mean() * 100.0),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
        "r2": float(1.0 - residual / max(total, 1e-12)),
    }


def circular_mae_deg(predicted_sin_cos: np.ndarray, true_phi_deg: np.ndarray) -> float:
    predicted = np.arctan2(predicted_sin_cos[:, 0], predicted_sin_cos[:, 1])
    true = np.deg2rad(true_phi_deg)
    difference = np.arctan2(np.sin(predicted - true), np.cos(predicted - true))
    return float(np.rad2deg(np.abs(difference)).mean())


def split_partitions(
    raw: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]], minimum: float, maximum: float
) -> dict[str, tuple[str, np.ndarray, pd.DataFrame]]:
    result: dict[str, tuple[str, np.ndarray, pd.DataFrame]] = {}

    def add(source: str, name: str, lower: float | None, upper: float | None) -> None:
        shape, _, frame = raw[source]
        mask = np.ones(len(frame), dtype=bool)
        if lower is not None:
            mask &= frame["rpm"].to_numpy() >= lower
        if upper is not None:
            mask &= frame["rpm"].to_numpy() <= upper
        result[name] = (source, shape[mask], frame.loc[mask].reset_index(drop=True))

    add("train", "train_core", minimum, maximum)
    add("val", "val_core", minimum, maximum)
    for source in ("test_id", "test_geometry", "test_hard"):
        add(source, f"{source}_core", minimum, maximum)
        add(source, f"{source}_ood_low", None, minimum - 1e-6)
        add(source, f"{source}_ood_high", maximum + 1e-6, None)
    return result


def make_psd_loader(
    shape: np.ndarray, frame: pd.DataFrame, frequency_mean: np.ndarray, frequency_std: np.ndarray,
    batch_size: int, shuffle: bool,
) -> DataLoader:
    frequency, geometry = targets_from_frame(frame)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(shape),
            torch.from_numpy(frame["rpm"].to_numpy(dtype=np.float32)),
            torch.from_numpy((frequency - frequency_mean) / frequency_std),
            torch.from_numpy(geometry),
            torch.from_numpy(shape[:, ::PSD_STRIDE]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def forward_model(
    code: str, model: nn.Module, signal: torch.Tensor, frequency_axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if code == "A4":
        assert isinstance(model, GeometryCombCNN)
        raw, scene = model(signal)
        rpm, four_frequency, geometry = physical_four_points(raw)
        comb_raw = model.decode_comb(geometry, scene)
        psd, amplitude = comb_forward(rpm / 60.0, comb_raw, frequency_axis)
        return rpm, four_frequency, geometry, psd, amplitude
    raw = model(signal)
    rpm, four_frequency, geometry = physical_four_points(raw)
    return rpm, four_frequency, geometry, None, None


def evaluate_array(
    code: str, model: nn.Module, shape: np.ndarray, frame: pd.DataFrame,
    frequency_mean: np.ndarray, frequency_std: np.ndarray, frequency_axis: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    loader = make_psd_loader(shape, frame, frequency_mean, frequency_std, 256, False)
    rpms, frequencies, geometries, psds = [], [], [], []
    with torch.no_grad():
        for signal, _, _, _, _ in loader:
            rpm, frequency, geometry, psd, _ = forward_model(code, model, signal.to(device), frequency_axis)
            rpms.append(rpm.cpu().numpy())
            frequencies.append(frequency.cpu().numpy())
            geometries.append(geometry.cpu().numpy())
            if psd is not None:
                psds.append(psd.cpu().numpy())
    return np.concatenate(rpms), np.concatenate(frequencies), np.concatenate(geometries), (np.concatenate(psds) if psds else None)


def evaluate_time(
    model: nn.Module, data_root: Path, source: str, frame: pd.DataFrame,
    frequency_axis: torch.Tensor, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(MMapWaveformDataset(data_root, source, frame, False), batch_size=128, shuffle=False)
    rpms, frequencies, geometries = [], [], []
    with torch.no_grad():
        for signal, _, _, _ in loader:
            rpm, frequency, geometry, _, _ = forward_model("A0", model, signal.to(device), frequency_axis)
            rpms.append(rpm.cpu().numpy())
            frequencies.append(frequency.cpu().numpy())
            geometries.append(geometry.cpu().numpy())
    return np.concatenate(rpms), np.concatenate(frequencies), np.concatenate(geometries)


def train_one(
    code: str, data_root: Path, partitions: dict[str, tuple[str, np.ndarray, pd.DataFrame]],
    frequency_mean: np.ndarray, frequency_std: np.ndarray, output: Path,
    epochs: int, seed: int, device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame]:
    set_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    model: nn.Module = GeometryCombCNN() if code == "A4" else PhysicalOutputCNN()
    model = model.to(device)
    frequency_axis = torch.arange(20.0, 1600.0 + 1e-4, float(PSD_STRIDE), device=device)
    train_source, train_shape, train_frame = partitions["train_core"]
    if code == "A0":
        loader = DataLoader(
            MMapWaveformDataset(data_root, train_source, train_frame, True),
            batch_size=72, shuffle=True, num_workers=0,
        )
    else:
        loader = make_psd_loader(train_shape, train_frame, frequency_mean, frequency_std, 72, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    frequency_mean_tensor = torch.from_numpy(frequency_mean).to(device)
    frequency_std_tensor = torch.from_numpy(frequency_std).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae, patience = math.inf, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        sums = {"total": 0.0, "rpm": 0.0, "geometry": 0.0, "four": 0.0, "psd": 0.0}
        count = 0
        for batch in loader:
            if code == "A0":
                signal, rpm_target, four_target_raw, geometry_target = batch
                psd_target = None
                four_target = (four_target_raw - torch.from_numpy(frequency_mean)) / torch.from_numpy(frequency_std)
            else:
                signal, rpm_target, four_target, geometry_target, psd_target = batch
                signal = signal + 0.012 * torch.randn_like(signal)
            signal = signal.to(device)
            rpm, frequency, geometry, reconstructed, amplitude = forward_model(code, model, signal, frequency_axis)
            rpm_loss = F.smooth_l1_loss(rpm / 1000.0, rpm_target.to(device) / 1000.0)
            geometry_loss = F.smooth_l1_loss(geometry, geometry_target.to(device))
            four_loss = F.smooth_l1_loss(
                (frequency - frequency_mean_tensor) / frequency_std_tensor, four_target.to(device)
            )
            psd_loss = torch.zeros((), device=device)
            if reconstructed is not None and psd_target is not None:
                psd_loss = F.smooth_l1_loss(reconstructed, psd_target.to(device))
            if code in {"A0", "A1"}:
                loss = rpm_loss
            elif code == "A2":
                loss = rpm_loss + 0.05 * geometry_loss
            elif code == "A3":
                loss = rpm_loss + 0.05 * geometry_loss + 0.08 * four_loss
            else:
                assert amplitude is not None
                loss = rpm_loss + 0.05 * geometry_loss + 0.08 * four_loss + 0.25 * psd_loss + 0.0005 * amplitude.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            batch_size = len(signal)
            count += batch_size
            sums["total"] += float(loss.detach()) * batch_size
            sums["rpm"] += float(rpm_loss.detach()) * batch_size
            sums["geometry"] += float(geometry_loss.detach()) * batch_size
            sums["four"] += float(four_loss.detach()) * batch_size
            sums["psd"] += float(psd_loss.detach()) * batch_size
        scheduler.step()
        val_source, val_shape, val_frame = partitions["val_core"]
        if code == "A0":
            val_prediction, _, _ = evaluate_time(model, data_root, val_source, val_frame, frequency_axis, device)
        else:
            val_prediction, _, _, _ = evaluate_array(
                code, model, val_shape, val_frame, frequency_mean, frequency_std, frequency_axis, device
            )
        val_mae = regression_metrics(val_frame["rpm"].to_numpy(dtype=np.float32), val_prediction)["mae_rpm"]
        history.append({
            "epoch": epoch + 1, "train_total_loss": sums["total"] / count,
            "train_rpm_loss": sums["rpm"] / count, "train_geometry_loss": sums["geometry"] / count,
            "train_four_point_loss": sums["four"] / count, "train_psd_loss": sums["psd"] / count,
            "validation_mae_rpm": val_mae, "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if val_mae < best_mae:
            best_mae, patience = val_mae, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 14:
                break
        if (epoch + 1) % 10 == 0:
            print(f"{code} epoch {epoch + 1}: validation MAE {val_mae:.2f} RPM", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    torch.save({"model_state": model.state_dict(), "model_code": code}, output / "best_model.pt")
    report: dict[str, object] = {
        "model_code": code, "model_name": MODEL_NAMES[code], "best_validation_mae_rpm": best_mae,
        "epochs_completed": len(history), "results": {},
    }
    saved_rows: list[pd.DataFrame] = []
    for split, (source, shape, frame) in partitions.items():
        if code == "A0":
            prediction, frequency_prediction, geometry_prediction = evaluate_time(
                model, data_root, source, frame, frequency_axis, device
            )
            psd_prediction = None
        else:
            prediction, frequency_prediction, geometry_prediction, psd_prediction = evaluate_array(
                code, model, shape, frame, frequency_mean, frequency_std, frequency_axis, device
            )
        actual = frame["rpm"].to_numpy(dtype=np.float32)
        frequency, geometry = targets_from_frame(frame)
        diagnostics: dict[str, float] = {}
        if code in {"A2", "A3", "A4"}:
            diagnostics.update({
                "gamma_mae_deg": float(np.abs(geometry_prediction[:, 0] * 75.0 - geometry[:, 0] * 75.0).mean()),
                "d_mae_mm": float(np.abs(geometry_prediction[:, 1] * 12.0 - geometry[:, 1] * 12.0).mean()),
                "phi_circular_mae_deg": circular_mae_deg(geometry_prediction[:, 2:4], frame["phi_deg"].to_numpy()),
            })
        if code in {"A3", "A4"}:
            diagnostics["four_point_frequency_mae_hz"] = float(np.abs(frequency_prediction - frequency).mean())
        if psd_prediction is not None:
            diagnostics["psd_shape_mae"] = float(np.abs(psd_prediction - shape[:, ::PSD_STRIDE]).mean())
        report["results"][split] = {"samples": int(len(actual)), **regression_metrics(actual, prediction), **diagnostics}
        rows = frame[["rpm", "severity", "gamma_deg", "phi_deg", "d_mm", "record_index"]].copy()
        rows["split"] = split
        rows["prediction_rpm"] = prediction
        rows["absolute_error_rpm"] = np.abs(prediction - actual)
        rows["model_code"] = code
        saved_rows.append(rows)
    predictions = pd.concat(saved_rows, ignore_index=True)
    predictions.to_csv(output / "per_sample_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, predictions


def main() -> None:
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=parent / "data" / "rde_v3_stratified_18000")
    parser.add_argument("--output-dir", type=Path, default=parent / "results" / "rde_v3_ablation_suite")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_NAMES), default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--train-rpm-min", type=float, default=700.0)
    parser.add_argument("--train-rpm-max", type=float, default=2400.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    raw = {split: spectral_features(args.data_root, split, 20.0, 1600.0) for split in SPLITS}
    partitions = split_partitions(raw, args.train_rpm_min, args.train_rpm_max)
    train_frame = partitions["train_core"][2]
    train_frequency, _ = targets_from_frame(train_frame)
    frequency_mean = train_frequency.mean(axis=0).astype(np.float32)
    frequency_std = np.maximum(train_frequency.std(axis=0), 1e-5).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "models": MODEL_NAMES, "requested_models": args.models, "seed": args.seed,
        "epochs_max": args.epochs, "device": str(device),
        "train_rpm_interval": [args.train_rpm_min, args.train_rpm_max],
        "low_ood_interval": [400.0, args.train_rpm_min],
        "high_ood_interval": [args.train_rpm_max, 3000.0],
        "input_policy": "A0 uses stored time waveform; A1-A4 use only 20-1600 Hz normalized log-PSD; no seven scalar summaries.",
        "checkpoint_policy": "Minimum val_core RPM MAE only; all test partitions untouched.",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    reports: dict[str, object] = {}
    all_predictions = []
    for code in args.models:
        print(f"Starting {code}: {MODEL_NAMES[code]}", flush=True)
        report, predictions = train_one(
            code, args.data_root, partitions, frequency_mean, frequency_std,
            args.output_dir / code, args.epochs, args.seed, device,
        )
        reports[code] = report
        all_predictions.append(predictions)
        print(f"Finished {code}", flush=True)
    (args.output_dir / "all_metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    pd.concat(all_predictions, ignore_index=True).to_csv(args.output_dir / "all_predictions.csv", index=False)


if __name__ == "__main__":
    main()
