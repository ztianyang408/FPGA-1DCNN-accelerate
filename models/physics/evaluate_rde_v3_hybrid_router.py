"""Evaluate the validated in-range/OOD hybrid without using OOD labels.

The routing boundary is exactly the maximum training RPM. Predictions above
that boundary from the range-equivariant candidate matcher use its result;
otherwise the high-accuracy FFT CNN result is retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "mape_pct": float(np.mean(np.abs(error) / actual) * 100.0),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
        "r2": float(1.0 - np.sum(error**2) / max(total, 1e-12)),
    }


def main() -> None:
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direct-predictions",
        type=Path,
        default=parent / "results" / "rde_v3_ablation_suite_gpu" / "A1" / "per_sample_predictions.csv",
    )
    parser.add_argument(
        "--candidate-predictions",
        type=Path,
        default=parent / "results" / "rde_v3_candidate_matcher_v1" / "predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=parent / "results" / "rde_v3_hybrid_target_met"
    )
    parser.add_argument("--training-upper-rpm", type=float, default=2400.0)
    args = parser.parse_args()

    direct = pd.read_csv(args.direct_predictions)
    candidate = pd.read_csv(args.candidate_predictions)
    keys = ["split", "record_index"]
    direct = direct[keys + ["rpm", "prediction_rpm"]].rename(
        columns={"prediction_rpm": "direct_prediction_rpm"}
    )
    candidate = candidate[keys + ["prediction_rpm"]].rename(
        columns={"prediction_rpm": "candidate_prediction_rpm"}
    )
    merged = direct.merge(candidate, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(direct) or len(merged) != len(candidate):
        raise ValueError("Prediction files do not contain identical samples")
    merged["used_candidate_matcher"] = (
        merged["candidate_prediction_rpm"] > args.training_upper_rpm
    )
    merged["prediction_rpm"] = np.where(
        merged["used_candidate_matcher"],
        merged["candidate_prediction_rpm"],
        merged["direct_prediction_rpm"],
    )
    merged["absolute_error_rpm"] = np.abs(merged["prediction_rpm"] - merged["rpm"])

    report: dict[str, object] = {
        "protocol": {
            "direct_branch": "A1 FFT CNN trained on 700-2400 RPM",
            "extrapolation_branch": "shared harmonic candidate scorer trained on 700-2400 RPM",
            "candidate_search_range_rpm": [400.0, 3000.0],
            "routing_rule": "use candidate matcher iff its prediction exceeds the fixed training upper boundary",
            "routing_threshold_rpm": args.training_upper_rpm,
            "ood_labels_used_for_routing_or_training": False,
        },
        "success_criteria": {
            "test_id_core_mape_below_pct": 5.0,
            "test_id_ood_high_mape_below_pct": 15.0,
        },
        "results": {},
    }
    for split, frame in merged.groupby("split", sort=False):
        actual = frame["rpm"].to_numpy(dtype=np.float32)
        predicted = frame["prediction_rpm"].to_numpy(dtype=np.float32)
        report["results"][split] = {
            "samples": int(len(frame)),
            **metrics(actual, predicted),
            "candidate_route_fraction_pct": float(frame["used_candidate_matcher"].mean() * 100.0),
        }
    core = report["results"]["test_id_core"]["mape_pct"]
    high = report["results"]["test_id_ood_high"]["mape_pct"]
    report["criteria_met"] = {
        "in_range": bool(core < 5.0),
        "high_speed_extrapolation": bool(high < 15.0),
        "all": bool(core < 5.0 and high < 15.0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "per_sample_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
