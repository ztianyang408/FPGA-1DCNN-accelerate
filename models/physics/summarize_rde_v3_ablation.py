"""Build tables and figures for the RDE v3 ablation presentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "A0": "A0 Time CNN",
    "A1": "A1 FFT CNN",
    "A2": "A2 + Geometry",
    "A3": "A3 + Eq. (10)",
    "A4": "A4 + Spectrum consistency",
}
KEY_SPLITS = ["test_id_core", "test_geometry_core", "test_hard_core", "test_id_ood_low", "test_id_ood_high"]
SPLIT_LABELS = ["ID generalization", "Unseen geometry", "Hard conditions", "Low-speed OOD", "High-speed OOD"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.results_dir / "summary"
    output.mkdir(parents=True, exist_ok=True)
    reports = json.loads((args.results_dir / "all_metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.results_dir / "all_predictions.csv")

    rows = []
    for code, report in reports.items():
        for split, metrics in report["results"].items():
            rows.append({"model_code": code, "model_name": MODEL_LABELS[code], "split": split, **metrics})
    long_metrics = pd.DataFrame(rows)
    long_metrics.to_csv(output / "all_split_metrics.csv", index=False)

    key = long_metrics[long_metrics["split"].isin(KEY_SPLITS)].copy()
    key["split"] = pd.Categorical(key["split"], categories=KEY_SPLITS, ordered=True)
    key = key.sort_values(["model_code", "split"])
    key.to_csv(output / "ppt_core_metrics_long.csv", index=False)
    for metric in ("mae_rpm", "mape_pct", "rmse_rpm", "bias_rpm", "r2"):
        key.pivot(index="model_name", columns="split", values=metric).to_csv(output / f"ppt_{metric}_table.csv")

    # Severity analysis uses the predefined core and OOD partitions without
    # pooling their very different RPM ranges.
    severity = (
        predictions.groupby(["model_code", "split", "severity"], observed=True)
        .apply(
            lambda frame: pd.Series(
                {
                    "samples": len(frame),
                    "mae_rpm": frame["absolute_error_rpm"].mean(),
                    "mape_pct": (frame["absolute_error_rpm"] / frame["rpm"]).mean() * 100.0,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    severity.to_csv(output / "severity_metrics.csv", index=False)

    comparison = []
    for split in KEY_SPLITS:
        values = key[key["split"] == split].set_index("model_code")
        comparison.append(
            {
                "split": split,
                "A4_vs_A1_mape_change_pct": 100.0 * (values.loc["A4", "mape_pct"] / values.loc["A1", "mape_pct"] - 1.0),
                "A4_vs_A3_mape_change_pct": 100.0 * (values.loc["A4", "mape_pct"] / values.loc["A3", "mape_pct"] - 1.0),
                "A4_vs_A1_mae_change_pct": 100.0 * (values.loc["A4", "mae_rpm"] / values.loc["A1", "mae_rpm"] - 1.0),
                "A4_vs_A3_mae_change_pct": 100.0 * (values.loc["A4", "mae_rpm"] / values.loc["A3", "mae_rpm"] - 1.0),
            }
        )
    pd.DataFrame(comparison).to_csv(output / "model_improvement_percentages.csv", index=False)

    colors = ["#6B7280", "#2F6B9A", "#4E9F7A", "#C58A2C", "#B64B4B"]
    x = np.arange(len(SPLIT_LABELS))
    width = 0.16
    figure, axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    for index, code in enumerate(MODEL_LABELS):
        values = [float(key[(key.model_code == code) & (key.split == split)]["mape_pct"].iloc[0]) for split in KEY_SPLITS]
        axis.bar(x + (index - 2) * width, values, width, label=MODEL_LABELS[code], color=colors[index])
    axis.set_ylabel("MAPE (%)")
    axis.set_xticks(x, SPLIT_LABELS)
    axis.set_title("Ablation study: generalization and speed extrapolation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=3, fontsize=9)
    figure.savefig(output / "ppt_ablation_mape.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    selected = ["test_id_core", "test_geometry_core", "test_hard_core"]
    for index, code in enumerate(MODEL_LABELS):
        values = [float(key[(key.model_code == code) & (key.split == split)]["mae_rpm"].iloc[0]) for split in selected]
        axes[0].plot(["ID", "Geometry", "Hard"], values, marker="o", linewidth=2, color=colors[index], label=MODEL_LABELS[code])
    axes[0].set(title="In-range generalization", ylabel="MAE (RPM)")
    axes[0].grid(alpha=0.25)
    for index, code in enumerate(MODEL_LABELS):
        values = [float(key[(key.model_code == code) & (key.split == split)]["mae_rpm"].iloc[0]) for split in ["test_id_ood_low", "test_id_ood_high"]]
        axes[1].plot(["Low OOD", "High OOD"], values, marker="o", linewidth=2, color=colors[index], label=MODEL_LABELS[code])
    axes[1].set(title="Speed extrapolation", ylabel="MAE (RPM)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper left")
    figure.savefig(output / "ppt_generalization_vs_ood.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    for index, code in enumerate(MODEL_LABELS):
        history = pd.read_csv(args.results_dir / code / "training_history.csv")
        axis.plot(history["epoch"], history["validation_mae_rpm"], linewidth=1.8, color=colors[index], label=MODEL_LABELS[code])
    axis.set(title="Validation convergence", xlabel="Epoch", ylabel="Validation MAE (RPM)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output / "ppt_validation_curves.png", dpi=200)
    plt.close(figure)

    summary = {
        "best_in_range_model": key[key.split == "test_id_core"].sort_values("mae_rpm").iloc[0]["model_code"],
        "best_low_ood_model": key[key.split == "test_id_ood_low"].sort_values("mae_rpm").iloc[0]["model_code"],
        "best_high_ood_model": key[key.split == "test_id_ood_high"].sort_values("mae_rpm").iloc[0]["model_code"],
        "caution": "A4 improves over A3 after adding spectrum consistency, but it does not dominate every baseline. Interpret the study as an accuracy-interpretability-extrapolation tradeoff.",
    }
    (output / "ppt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
