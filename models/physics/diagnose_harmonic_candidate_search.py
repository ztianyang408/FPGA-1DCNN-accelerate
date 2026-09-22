"""Evaluate label-free harmonic candidate scores on the fixed RDE v3 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_rde_v3_spectral_cnn import spectral_features


def candidate_scores(psd: np.ndarray, frequency: np.ndarray, candidates: np.ndarray, mode: str) -> np.ndarray:
    orders = np.arange(1, 65, dtype=np.float32)
    query = candidates[:, None] * orders[None]
    valid = query <= frequency[-1]
    result = np.empty((len(psd), len(candidates)), dtype=np.float32)
    for index, spectrum in enumerate(psd):
        values = np.interp(query.ravel(), frequency, spectrum, left=-4.0, right=-4.0).reshape(query.shape)
        values[~valid] = -4.0
        if mode == "weighted_mean":
            weight = valid / np.sqrt(orders)[None]
            result[index] = np.sum(values * weight, axis=1) / np.maximum(weight.sum(axis=1), 1e-6)
        elif mode == "topk":
            values[~valid] = -20.0
            result[index] = np.mean(np.sort(values, axis=1)[:, -10:], axis=1)
        elif mode == "contrast":
            # True harmonics should be local maxima, unlike a smooth broadband background.
            left = np.interp((query - 0.45).ravel(), frequency, spectrum, left=-4.0, right=-4.0).reshape(query.shape)
            right = np.interp((query + 0.45).ravel(), frequency, spectrum, left=-4.0, right=-4.0).reshape(query.shape)
            contrast = values - 0.5 * (left + right)
            contrast[~valid] = -20.0
            result[index] = np.mean(np.sort(contrast, axis=1)[:, -12:], axis=1)
        else:
            raise ValueError(mode)
    return result


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "mape_pct": float(np.mean(np.abs(error) / actual) * 100),
        "bias_rpm": float(error.mean()),
        "r2": float(1.0 - np.sum(error**2) / max(total, 1e-12)),
    }


def main() -> None:
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=parent / "data" / "rde_v3_stratified_18000")
    parser.add_argument("--output", type=Path, default=parent / "results" / "harmonic_candidate_diagnostic.json")
    args = parser.parse_args()
    candidates_rpm = np.arange(400.0, 3000.01, 5.0, dtype=np.float32)
    candidates_hz = candidates_rpm / 60.0
    frequency = np.arange(20.0, 1600.01, 1.0, dtype=np.float32)
    report: dict[str, object] = {}
    for split in ("val", "test_id", "test_geometry", "test_hard"):
        psd, _, frame = spectral_features(args.data_root, split, 20.0, 1600.0)
        rpm = frame["rpm"].to_numpy(dtype=np.float32)
        partitions = {
            "core": (rpm >= 700) & (rpm <= 2400),
            "ood_low": rpm < 700,
            "ood_high": rpm > 2400,
        }
        report[split] = {}
        for mode in ("weighted_mean", "topk", "contrast"):
            score = candidate_scores(psd, frequency, candidates_hz, mode)
            prediction = candidates_rpm[np.argmax(score, axis=1)]
            report[split][mode] = {
                name: {"samples": int(mask.sum()), **metrics(rpm[mask], prediction[mask])}
                for name, mask in partitions.items()
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
