"""Evaluate a prediction ensemble without refitting on any test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - true
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(Path(run) / "per_sample_predictions.csv") for run in args.runs]
    keys = ["split", "file"]
    base = frames[0].copy()
    prediction_columns = ("black_box_prediction_rpm", "physics_guided_prediction_rpm")
    result: dict[str, object] = {"runs": args.runs, "results": {}}

    for column in prediction_columns:
        values = []
        for frame in frames:
            aligned = base[keys].merge(frame[keys + [column]], on=keys, how="left", validate="one_to_one")
            if aligned[column].isna().any():
                raise ValueError(f"Missing aligned prediction for {column}")
            values.append(aligned[column].to_numpy(dtype=np.float32))
        stacked = np.stack(values)
        name = column.removesuffix("_prediction_rpm")
        base[f"{name}_ensemble_prediction_rpm"] = stacked.mean(axis=0)
        base[f"{name}_ensemble_std_rpm"] = stacked.std(axis=0)

    for split, frame in base.groupby("split", sort=False):
        true = frame["rpm_label"].to_numpy(dtype=np.float32)
        result["results"][split] = {
            "samples": int(len(frame)),
            "black_box_ensemble": metrics(true, frame["black_box_ensemble_prediction_rpm"].to_numpy()),
            "physics_guided_ensemble": metrics(
                true, frame["physics_guided_ensemble_prediction_rpm"].to_numpy()
            ),
            "black_box_mean_model_std_rpm": float(frame["black_box_ensemble_std_rpm"].mean()),
        }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base.to_csv(output / "per_sample_ensemble_predictions.csv", index=False)
    (output / "ensemble_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
