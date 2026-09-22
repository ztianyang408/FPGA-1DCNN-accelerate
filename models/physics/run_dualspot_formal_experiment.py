"""Formal dual-spot RPM extrapolation experiment.

The data package supplies dual ordinary-PD voltage traces and simulated
geometry labels. This script keeps its published train/validation/test splits
intact and compares:

1. A black-box dual-channel PSD regressor.
2. A physics-guided dual-spot inverse model. It predicts RPM and geometry,
   uses common/differential spectra from opposite illumination positions, and
   enforces the known opposite-spot azimuth relation.

This is deliberately not called a full PINN: the package does not contain the
arbitrary-incidence optical forward equation required to reconstruct a PSD from
geometry alone. The geometry supervision is available only because this is a
simulated calibration package; it can be removed when moving to real data.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dualspot_rde_forward import DualSpotRDEForward, local_rde_frequencies_numpy


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate_hz: float = 12_000.0
    min_frequency_hz: float = 15.0
    max_frequency_hz: float = 1_200.0
    frequency_stride: int = 4
    welch_length: int = 1024
    welch_hop: int = 512


GEOMETRY_COLUMNS = (
    "gamma_deg",
    "d_theta0",
    "d_thetapi",
    "ring_radius",
    "phi0_sin",
    "phi0_cos",
    "phipi_sin",
    "phipi_cos",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_features(row: pd.Series, data_root: Path, config: FeatureConfig) -> np.ndarray:
    frame = pd.read_csv(data_root / row["file"])
    raw_signal = frame[["theta0_v", "theta_pi_v"]].to_numpy(dtype=np.float32).T
    raw_signal = raw_signal - raw_signal.mean(axis=1, keepdims=True)
    signal_rms = np.sqrt(np.mean(raw_signal**2, axis=1) + 1e-12)
    signal = raw_signal / signal_rms[:, None]

    spectrum = np.abs(np.fft.rfft(signal, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(signal.shape[1], d=1.0 / config.sample_rate_hz)
    mask = (frequencies >= config.min_frequency_hz) & (frequencies <= config.max_frequency_hz)
    band_spectrum = spectrum[:, mask]
    band_frequencies = frequencies[mask]
    log_psd = np.log(band_spectrum + 1e-7)[:, :: config.frequency_stride]
    log_psd = (log_psd - log_psd.mean(axis=1, keepdims=True)) / (
        log_psd.std(axis=1, keepdims=True) + 1e-7
    )

    # Opposite illumination positions: common spectra carry shared RPM content,
    # while differential spectra carry geometry-sensitive content.
    common = 0.5 * (log_psd[0] + log_psd[1])
    differential = 0.5 * (log_psd[0] - log_psd[1])

    # A single FFT cross spectrum has unit coherence by construction. Welch
    # averaging makes coherence a meaningful measure of shared dual-channel
    # content and its phase retains the relative timing information.
    window = np.hanning(config.welch_length).astype(np.float32)
    starts = range(0, signal.shape[1] - config.welch_length + 1, config.welch_hop)
    auto0: list[np.ndarray] = []
    auto1: list[np.ndarray] = []
    cross: list[np.ndarray] = []
    for start in starts:
        segment = signal[:, start : start + config.welch_length] * window
        fft = np.fft.rfft(segment, axis=1)
        auto0.append(np.abs(fft[0]) ** 2)
        auto1.append(np.abs(fft[1]) ** 2)
        cross.append(fft[0] * np.conj(fft[1]))
    welch0 = np.mean(auto0, axis=0)
    welch1 = np.mean(auto1, axis=0)
    welch_cross = np.mean(cross, axis=0)
    welch_frequencies = np.fft.rfftfreq(config.welch_length, d=1.0 / config.sample_rate_hz)
    welch_mask = (welch_frequencies >= config.min_frequency_hz) & (
        welch_frequencies <= config.max_frequency_hz
    )
    coherence = np.abs(welch_cross) ** 2 / (welch0 * welch1 + 1e-12)
    cross_phase = np.angle(welch_cross)
    coherence = coherence[welch_mask][:: config.frequency_stride]
    cross_phase = cross_phase[welch_mask][:: config.frequency_stride]

    # Do not normalize these quantities per waveform: they tell the network
    # whether spectral differences are a trustworthy rotational signature or
    # likely dominated by noise, fading, or channel imbalance.
    total_power = np.sum(band_spectrum, axis=1) + 1e-12
    normalized_band_power = band_spectrum / total_power[:, None]
    centroid = np.sum(normalized_band_power * band_frequencies, axis=1)
    bandwidth = np.sqrt(
        np.sum(normalized_band_power * (band_frequencies - centroid[:, None]) ** 2, axis=1)
    )
    peak_index = np.argmax(band_spectrum, axis=1)
    peak_frequency = band_frequencies[peak_index]
    peak_prominence = band_spectrum[np.arange(2), peak_index] / (
        np.median(band_spectrum, axis=1) + 1e-12
    )
    spectral_entropy = -np.sum(
        normalized_band_power * np.log(normalized_band_power + 1e-12), axis=1
    ) / np.log(normalized_band_power.shape[1])
    scalar_features = np.concatenate(
        [
            np.log(signal_rms + 1e-12),
            np.log(total_power),
            np.array([np.log(signal_rms[0] / signal_rms[1]), np.log(total_power[0] / total_power[1])]),
            centroid,
            bandwidth,
            peak_frequency,
            np.log(peak_prominence + 1e-12),
            spectral_entropy,
            np.array([np.mean(coherence), np.max(coherence)]),
        ]
    )
    return np.concatenate(
        [common, differential, coherence, np.sin(cross_phase), np.cos(cross_phase), scalar_features]
    ).astype(np.float32)


def geometry_targets(frame: pd.DataFrame) -> np.ndarray:
    phi0 = np.deg2rad(frame["phi_theta0_deg"].to_numpy(dtype=np.float32))
    phipi = np.deg2rad(frame["phi_thetapi_deg"].to_numpy(dtype=np.float32))
    return np.column_stack(
        [
            frame["gamma_deg"].to_numpy(dtype=np.float32),
            frame["d_theta0"].to_numpy(dtype=np.float32),
            frame["d_thetapi"].to_numpy(dtype=np.float32),
            frame["ring_radius"].to_numpy(dtype=np.float32),
            np.sin(phi0),
            np.cos(phi0),
            np.sin(phipi),
            np.cos(phipi),
        ]
    ).astype(np.float32)


def frequency_targets(frame: pd.DataFrame) -> np.ndarray:
    return frame[
        [
            "theta0_freq_p05_hz",
            "theta0_freq_p50_hz",
            "theta0_freq_p95_hz",
            "theta_pi_freq_p05_hz",
            "theta_pi_freq_p50_hz",
            "theta_pi_freq_p95_hz",
        ]
    ].to_numpy(dtype=np.float32)


def load_split(
    manifest: pd.DataFrame, split: str, data_root: Path, feature_config: FeatureConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    subset = manifest.loc[manifest["split"] == split].reset_index(drop=True)
    features = np.stack([read_features(row, data_root, feature_config) for _, row in subset.iterrows()])
    rpm = subset["rpm_label"].to_numpy(dtype=np.float32)
    geometry = geometry_targets(subset)
    return features, rpm, geometry, frequency_targets(subset), subset


def normalize_features(
    train_features: np.ndarray, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use train-only feature scaling; retain relative quality information."""
    mean = train_features.mean(axis=0)
    std = np.maximum(train_features.std(axis=0), 1e-5)
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def oracle_forward_quantiles(frame: pd.DataFrame) -> np.ndarray:
    """Evaluate the vector RDE formula with the known synthetic geometry."""
    theta0 = local_rde_frequencies_numpy(
        frame["rpm_label"], frame["gamma_deg"], frame["d_theta0"], frame["phi_theta0_deg"], frame["ring_radius"], 0.0
    )
    theta_pi = local_rde_frequencies_numpy(
        frame["rpm_label"], frame["gamma_deg"], frame["d_thetapi"], frame["phi_thetapi_deg"], frame["ring_radius"], np.pi
    )
    q0 = np.quantile(np.abs(theta0), [0.05, 0.50, 0.95], axis=1).T
    qpi = np.quantile(np.abs(theta_pi), [0.05, 0.50, 0.95], axis=1).T
    return np.concatenate([q0, qpi], axis=1).astype(np.float32)


def fit_forward_calibration(raw: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-quantile affine calibration using the training split only."""
    scales = np.empty(raw.shape[1], dtype=np.float32)
    offsets = np.empty(raw.shape[1], dtype=np.float32)
    for index in range(raw.shape[1]):
        design = np.column_stack([raw[:, index], np.ones(len(raw))])
        scales[index], offsets[index] = np.linalg.lstsq(design, observed[:, index], rcond=None)[0]
    return scales, offsets


class SharedEncoder(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BlackBoxRegressor(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.encoder = SharedEncoder(inputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).squeeze(1)


class DualSpotPhysicsGuidedRegressor(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.encoder = SharedEncoder(inputs, 1 + len(GEOMETRY_COLUMNS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.encoder(x)
        return raw[:, 0], raw[:, 1:]


class StructuredPhysicsInversionRegressor(nn.Module):
    """Predict only observable frequency diagnostics and geometry, never RPM."""

    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.encoder = SharedEncoder(inputs, len(GEOMETRY_COLUMNS) + 6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.encoder(x)
        return raw[:, : len(GEOMETRY_COLUMNS)], raw[:, len(GEOMETRY_COLUMNS) :]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
    }


def evaluate_rpm(
    model: nn.Module,
    x: np.ndarray,
    rpm_mean: float,
    rpm_std: float,
    device: torch.device,
    physics: bool,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(x).to(device))
        normalized = output[0] if physics else output
    return (normalized.cpu().numpy() * rpm_std + rpm_mean).astype(np.float32)


def inversion_rpm_from_outputs(
    normalized_geometry: torch.Tensor,
    normalized_frequency: torch.Tensor,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    frequency_mean: torch.Tensor,
    frequency_std: torch.Tensor,
    calibration_scale: torch.Tensor,
    calibration_offset: torch.Tensor,
    forward_layer: DualSpotRDEForward,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert calibrated forward quantiles with a differentiable LS solution."""
    geometry = normalized_geometry * geometry_std + geometry_mean
    phi0_deg = torch.rad2deg(torch.atan2(geometry[:, 4], geometry[:, 5]))
    phipi_deg = torch.rad2deg(torch.atan2(geometry[:, 6], geometry[:, 7]))
    unit_rpm = torch.ones_like(phi0_deg)
    unit_quantiles = torch.cat(
        [
            forward_layer.quantiles(
                forward_layer.local_frequencies(unit_rpm, geometry[:, 0], geometry[:, 1], phi0_deg, geometry[:, 3], 0.0)
            ),
            forward_layer.quantiles(
                forward_layer.local_frequencies(unit_rpm, geometry[:, 0], geometry[:, 2], phipi_deg, geometry[:, 3], float(np.pi))
            ),
        ],
        dim=1,
    )
    slope = unit_quantiles * calibration_scale
    observed_frequency = normalized_frequency * frequency_std + frequency_mean
    rpm = torch.sum(slope * (observed_frequency - calibration_offset), dim=1) / torch.clamp(
        torch.sum(slope.square(), dim=1), min=1e-8
    )
    return rpm, geometry


def evaluate_structured_rpm(
    model: nn.Module,
    x: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    frequency_mean: np.ndarray,
    frequency_std: np.ndarray,
    forward_calibration: tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    forward_layer = DualSpotRDEForward().to(device)
    with torch.no_grad():
        geometry, frequency = model(torch.from_numpy(x).to(device))
        rpm, _ = inversion_rpm_from_outputs(
            geometry,
            frequency,
            torch.from_numpy(geometry_mean).to(device),
            torch.from_numpy(geometry_std).to(device),
            torch.from_numpy(frequency_mean).to(device),
            torch.from_numpy(frequency_std).to(device),
            torch.from_numpy(forward_calibration[0]).to(device),
            torch.from_numpy(forward_calibration[1]).to(device),
            forward_layer,
        )
    return rpm.cpu().numpy().astype(np.float32)


def geometry_metrics(
    predicted: np.ndarray, target: np.ndarray, geometry_std: np.ndarray
) -> dict[str, float]:
    # Values are decoded after z-score normalization. Sine/cosine dimensions
    # are intentionally left as an auxiliary diagnostic instead of degrees.
    physical = predicted[:, :4] * geometry_std[:4]
    target_physical = target[:, :4] * geometry_std[:4]
    return {
        "gamma_mae_deg": float(np.abs(physical[:, 0] - target_physical[:, 0]).mean()),
        "d_theta0_mae": float(np.abs(physical[:, 1] - target_physical[:, 1]).mean()),
        "d_thetapi_mae": float(np.abs(physical[:, 2] - target_physical[:, 2]).mean()),
        "ring_radius_mae": float(np.abs(physical[:, 3] - target_physical[:, 3]).mean()),
    }


def train_black_box(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, float, float]:
    rpm_mean = float(train_y.mean())
    rpm_std = float(train_y.std())
    model = BlackBoxRegressor(train_x.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy((train_y - rpm_mean) / rpm_std)),
        batch_size=64,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    patience = 0
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            prediction = model(x_batch.to(device))
            loss = F.smooth_l1_loss(prediction, y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_prediction = evaluate_rpm(model, val_x, rpm_mean, rpm_std, device, physics=False)
        val_mae = regression_metrics(val_y, val_prediction)["mae_rpm"]
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 28:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, rpm_mean, rpm_std


def train_physics_guided(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_geometry: np.ndarray,
    train_frequency: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    forward_calibration: tuple[np.ndarray, np.ndarray],
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, float, float, np.ndarray, np.ndarray, np.ndarray]:
    rpm_mean = float(train_y.mean())
    rpm_std = float(train_y.std())
    geometry_mean = train_geometry.mean(axis=0)
    geometry_std = train_geometry.std(axis=0)
    geometry_std = np.maximum(geometry_std, 1e-5)
    normalized_geometry = (train_geometry - geometry_mean) / geometry_std
    frequency_mean = train_frequency.mean(axis=0)
    frequency_std = np.maximum(train_frequency.std(axis=0), 1e-5)
    normalized_frequency = (train_frequency - frequency_mean) / frequency_std
    geometry_mean_tensor = torch.from_numpy(geometry_mean).to(device)
    geometry_std_tensor = torch.from_numpy(geometry_std).to(device)
    frequency_mean_tensor = torch.from_numpy(frequency_mean).to(device)
    frequency_std_tensor = torch.from_numpy(frequency_std).to(device)
    calibration_scale = torch.from_numpy(forward_calibration[0]).to(device)
    calibration_offset = torch.from_numpy(forward_calibration[1]).to(device)
    forward_layer = DualSpotRDEForward().to(device)

    model = DualSpotPhysicsGuidedRegressor(train_x.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy((train_y - rpm_mean) / rpm_std),
            torch.from_numpy(normalized_geometry),
            torch.from_numpy(normalized_frequency),
        ),
        batch_size=64,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    patience = 0

    for _ in range(epochs):
        model.train()
        for x_batch, rpm_batch, geometry_batch, frequency_batch in loader:
            rpm_prediction, geometry_prediction = model(x_batch.to(device))
            rpm_loss = F.smooth_l1_loss(rpm_prediction, rpm_batch.to(device))
            geometry_loss = F.smooth_l1_loss(geometry_prediction, geometry_batch.to(device))

            # phi_pi = phi_0 - pi for opposite illumination positions. This is
            # evaluated in decoded sine/cosine form and needs no extra label.
            decoded_geometry = geometry_prediction * geometry_std_tensor + geometry_mean_tensor
            phi0 = torch.atan2(decoded_geometry[:, 4], decoded_geometry[:, 5])
            phipi = torch.atan2(decoded_geometry[:, 6], decoded_geometry[:, 7])
            opposite_loss = torch.mean((torch.cos(phi0 - phipi) + 1.0).square())

            # Opposite spots should share the same nominal displacement; the
            # weak term retains the true sample-to-sample mismatch.
            displacement_balance = torch.mean((decoded_geometry[:, 1] - decoded_geometry[:, 2]).square())

            phi0_deg = torch.rad2deg(torch.atan2(decoded_geometry[:, 4], decoded_geometry[:, 5]))
            phipi_deg = torch.rad2deg(torch.atan2(decoded_geometry[:, 6], decoded_geometry[:, 7]))
            forward0 = forward_layer.local_frequencies(
                rpm_prediction * rpm_std + rpm_mean,
                decoded_geometry[:, 0],
                decoded_geometry[:, 1],
                phi0_deg,
                decoded_geometry[:, 3],
                0.0,
            )
            forward_pi = forward_layer.local_frequencies(
                rpm_prediction * rpm_std + rpm_mean,
                decoded_geometry[:, 0],
                decoded_geometry[:, 2],
                phipi_deg,
                decoded_geometry[:, 3],
                float(np.pi),
            )
            forward_quantiles = torch.cat(
                [forward_layer.quantiles(forward0), forward_layer.quantiles(forward_pi)], dim=1
            )
            calibrated_quantiles = forward_quantiles * calibration_scale + calibration_offset
            normalized_quantiles = (calibrated_quantiles - frequency_mean_tensor) / frequency_std_tensor
            forward_loss = F.smooth_l1_loss(normalized_quantiles, frequency_batch.to(device))
            loss = (
                rpm_loss
                + 0.15 * geometry_loss
                + 0.20 * forward_loss
                + 0.04 * opposite_loss
                + 0.01 * displacement_balance
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_prediction = evaluate_rpm(model, val_x, rpm_mean, rpm_std, device, physics=True)
        val_mae = regression_metrics(val_y, val_prediction)["mae_rpm"]
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 28:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, rpm_mean, rpm_std, geometry_std, frequency_mean, frequency_std


def train_structured_inversion(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_geometry: np.ndarray,
    train_frequency: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    forward_calibration: tuple[np.ndarray, np.ndarray],
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geometry_mean = train_geometry.mean(axis=0).astype(np.float32)
    geometry_std = np.maximum(train_geometry.std(axis=0), 1e-5).astype(np.float32)
    frequency_mean = train_frequency.mean(axis=0).astype(np.float32)
    frequency_std = np.maximum(train_frequency.std(axis=0), 1e-5).astype(np.float32)
    normalized_geometry = (train_geometry - geometry_mean) / geometry_std
    normalized_frequency = (train_frequency - frequency_mean) / frequency_std
    geometry_mean_tensor = torch.from_numpy(geometry_mean).to(device)
    geometry_std_tensor = torch.from_numpy(geometry_std).to(device)
    frequency_mean_tensor = torch.from_numpy(frequency_mean).to(device)
    frequency_std_tensor = torch.from_numpy(frequency_std).to(device)
    calibration_scale = torch.from_numpy(forward_calibration[0]).to(device)
    calibration_offset = torch.from_numpy(forward_calibration[1]).to(device)
    forward_layer = DualSpotRDEForward().to(device)
    model = StructuredPhysicsInversionRegressor(train_x.shape[1]).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
            torch.from_numpy(normalized_geometry),
            torch.from_numpy(normalized_frequency),
        ),
        batch_size=64,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    patience = 0
    for _ in range(epochs):
        model.train()
        for x_batch, rpm_batch, geometry_batch, frequency_batch in loader:
            geometry_prediction, frequency_prediction = model(x_batch.to(device))
            rpm_prediction, decoded_geometry = inversion_rpm_from_outputs(
                geometry_prediction,
                frequency_prediction,
                geometry_mean_tensor,
                geometry_std_tensor,
                frequency_mean_tensor,
                frequency_std_tensor,
                calibration_scale,
                calibration_offset,
                forward_layer,
            )
            rpm_loss = F.smooth_l1_loss(rpm_prediction / 400.0, rpm_batch.to(device) / 400.0)
            geometry_loss = F.smooth_l1_loss(geometry_prediction, geometry_batch.to(device))
            frequency_loss = F.smooth_l1_loss(frequency_prediction, frequency_batch.to(device))
            phi0 = torch.atan2(decoded_geometry[:, 4], decoded_geometry[:, 5])
            phipi = torch.atan2(decoded_geometry[:, 6], decoded_geometry[:, 7])
            opposite_loss = torch.mean((torch.cos(phi0 - phipi) + 1.0).square())
            loss = rpm_loss + 0.15 * geometry_loss + 0.30 * frequency_loss + 0.04 * opposite_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        val_prediction = evaluate_structured_rpm(
            model, val_x, geometry_mean, geometry_std, frequency_mean, frequency_std, forward_calibration, device
        )
        val_mae = regression_metrics(val_y, val_prediction)["mae_rpm"]
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 35:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, geometry_mean, geometry_std, frequency_mean, frequency_std


def predicted_geometry(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        _, geometry = model(torch.from_numpy(x).to(device))
    return geometry.cpu().numpy()


def predicted_structured_geometry(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        geometry, _ = model(torch.from_numpy(x).to(device))
    return geometry.cpu().numpy()


def save_scatter(output: Path, title: str, true: np.ndarray, predicted: np.ndarray) -> None:
    lower = float(min(true.min(), predicted.min()))
    upper = float(max(true.max(), predicted.max()))
    figure, axis = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    axis.scatter(true, predicted, s=17, alpha=0.72)
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axis.set(title=title, xlabel="True RPM", ylabel="Predicted RPM")
    axis.grid(alpha=0.25)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal dual-spot RPM extrapolation benchmark.")
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent / "data" / "dual_spot_theta0_pi_formal_extrap_csv_v1"),
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent / "results" / "dualspot_formal_experiment")
    )
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(data_root / "manifest.csv")
    feature_config = FeatureConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for split in ("train", "val", "test_in", "test_edge_low", "test_edge_high", "test_ood_low", "test_ood_high"):
        datasets[split] = load_split(manifest, split, data_root, feature_config)
        print(f"Loaded {split}: {len(datasets[split][1])} waveforms")

    # Feature scaling is fit once on training waveforms. Unlike the former
    # per-waveform-only preprocessing, scalar quality features remain present.
    raw_train_features = datasets["train"][0]
    normalized_datasets: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]
    ] = {}
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    for split, (features, rpm, geometry, frequency, frame) in datasets.items():
        normalized, mean, std = normalize_features(raw_train_features, features)
        normalized_datasets[split] = (normalized, rpm, geometry, frequency, frame)
        if split == "train":
            feature_mean, feature_std = mean, std
    datasets = normalized_datasets
    assert feature_mean is not None and feature_std is not None

    train_x, train_y, train_geometry, train_frequency, train_frame = datasets["train"]
    val_x, val_y, _, _, _ = datasets["val"]
    forward_calibration = fit_forward_calibration(oracle_forward_quantiles(train_frame), train_frequency)
    black_box, black_mean, black_std = train_black_box(
        train_x, train_y, val_x, val_y, args.epochs, device
    )
    guided, guided_mean, guided_std, geometry_std, frequency_mean, frequency_std = train_physics_guided(
        train_x,
        train_y,
        train_geometry,
        train_frequency,
        val_x,
        val_y,
        forward_calibration,
        args.epochs,
        device,
    )
    structured, structured_geometry_mean, structured_geometry_std, structured_frequency_mean, structured_frequency_std = train_structured_inversion(
        train_x,
        train_y,
        train_geometry,
        train_frequency,
        val_x,
        val_y,
        forward_calibration,
        args.epochs,
        device,
    )

    torch.save(
        {
            "model_state": black_box.state_dict(),
            "rpm_mean": black_mean,
            "rpm_std": black_std,
            "feature_config": asdict(feature_config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        output_dir / "black_box_best.pt",
    )
    torch.save(
        {
            "model_state": guided.state_dict(),
            "rpm_mean": guided_mean,
            "rpm_std": guided_std,
            "geometry_std": geometry_std,
            "geometry_columns": GEOMETRY_COLUMNS,
            "frequency_mean": frequency_mean,
            "frequency_std": frequency_std,
            "forward_calibration_scale": forward_calibration[0],
            "forward_calibration_offset": forward_calibration[1],
            "feature_config": asdict(feature_config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        output_dir / "physics_guided_best.pt",
    )
    torch.save(
        {
            "model_state": structured.state_dict(),
            "geometry_mean": structured_geometry_mean,
            "geometry_std": structured_geometry_std,
            "frequency_mean": structured_frequency_mean,
            "frequency_std": structured_frequency_std,
            "forward_calibration_scale": forward_calibration[0],
            "forward_calibration_offset": forward_calibration[1],
            "feature_config": asdict(feature_config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        output_dir / "structured_inversion_best.pt",
    )

    report: dict[str, object] = {
        "protocol": {
            "train": "600-1400 RPM only",
            "validation": "600-1400 RPM; used for checkpoint selection only",
            "test": "predefined test_in, edge, and OOD splits; not used during training",
            "physics_guided_model": "dual common/differential PSD representation + geometry multi-task supervision + opposite-spot constraints + calibrated arbitrary-incidence RDE frequency-distribution loss",
            "structured_inversion_model": "PSD-to-frequency-and-geometry network followed by a mandatory differentiable least-squares inversion of calibrated RDE quantiles; it contains no direct RPM output path",
            "forward_calibration": "six affine frequency-quantile calibration factors fit on the training split only",
            "limitation": "the full generator noise and scattering model is unavailable; PSD amplitude reconstruction is therefore not included",
        },
        "feature_config": asdict(feature_config),
        "feature_representation": "per-channel normalized PSD shape, dual-channel Welch coherence and phase, plus train-normalized absolute power and spectral quality scalars",
        "results": {},
    }
    all_true: list[np.ndarray] = []
    all_black: list[np.ndarray] = []
    all_guided: list[np.ndarray] = []
    all_structured: list[np.ndarray] = []
    per_sample: list[pd.DataFrame] = []
    for split, (x, rpm, geometry, frequency, frame) in datasets.items():
        black_prediction = evaluate_rpm(black_box, x, black_mean, black_std, device, physics=False)
        guided_prediction = evaluate_rpm(guided, x, guided_mean, guided_std, device, physics=True)
        structured_prediction = evaluate_structured_rpm(
            structured,
            x,
            structured_geometry_mean,
            structured_geometry_std,
            structured_frequency_mean,
            structured_frequency_std,
            forward_calibration,
            device,
        )
        geometry_prediction = predicted_geometry(guided, x, device)
        structured_geometry_prediction = predicted_structured_geometry(structured, x, device)
        report["results"][split] = {
            "samples": int(len(rpm)),
            "black_box": regression_metrics(rpm, black_prediction),
            "physics_guided": {
                **regression_metrics(rpm, guided_prediction),
                "geometry_diagnostics": geometry_metrics(
                    geometry_prediction, (geometry - train_geometry.mean(axis=0)) / geometry_std, geometry_std
                ),
                "oracle_forward_frequency_mae_hz": float(
                    np.abs(
                        oracle_forward_quantiles(frame) * forward_calibration[0] + forward_calibration[1]
                        - frequency
                    ).mean()
                ),
            },
            "structured_inversion": {
                **regression_metrics(rpm, structured_prediction),
                "geometry_diagnostics": geometry_metrics(
                    structured_geometry_prediction,
                    (geometry - structured_geometry_mean) / structured_geometry_std,
                    structured_geometry_std,
                ),
            },
        }
        prediction_frame = frame.copy()
        prediction_frame["black_box_prediction_rpm"] = black_prediction
        prediction_frame["physics_guided_prediction_rpm"] = guided_prediction
        prediction_frame["structured_inversion_prediction_rpm"] = structured_prediction
        prediction_frame["black_box_absolute_error_rpm"] = np.abs(black_prediction - rpm)
        prediction_frame["physics_guided_absolute_error_rpm"] = np.abs(guided_prediction - rpm)
        prediction_frame["structured_inversion_absolute_error_rpm"] = np.abs(structured_prediction - rpm)
        per_sample.append(prediction_frame)
        if split.startswith("test_"):
            all_true.append(rpm)
            all_black.append(black_prediction)
            all_guided.append(guided_prediction)
            all_structured.append(structured_prediction)

    all_true_array = np.concatenate(all_true)
    save_scatter(output_dir / "black_box_all_test.png", "Black-box: all predefined tests", all_true_array, np.concatenate(all_black))
    save_scatter(
        output_dir / "physics_guided_all_test.png",
        "Physics-guided: all predefined tests",
        all_true_array,
        np.concatenate(all_guided),
    )
    save_scatter(
        output_dir / "structured_inversion_all_test.png",
        "Structured physics inversion: all predefined tests",
        all_true_array,
        np.concatenate(all_structured),
    )
    pd.concat(per_sample, ignore_index=True).to_csv(output_dir / "per_sample_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
