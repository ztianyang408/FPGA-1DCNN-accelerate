from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "phase1" / "results"


def load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return pd.read_csv(metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 1 runs")
    parser.add_argument("--tree-run", type=Path, default=DEFAULT_RESULTS_ROOT / "20260829_112909_tree")
    parser.add_argument("--cnn-run", type=Path, default=DEFAULT_RESULTS_ROOT / "20260829_214336_cnn")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_ROOT / "phase1_comparison.json")
    args = parser.parse_args()

    tree = load_metrics(args.tree_run)
    cnn = load_metrics(args.cnn_run)
    merged = tree.merge(cnn, on="split", suffixes=("_tree", "_cnn"))
    merged["mae_delta_cnn_minus_tree"] = merged["mae_cnn"] - merged["mae_tree"]
    merged["rmse_delta_cnn_minus_tree"] = merged["rmse_cnn"] - merged["rmse_tree"]
    merged["mape_delta_cnn_minus_tree"] = merged["mape_cnn"] - merged["mape_tree"]
    merged["r2_delta_cnn_minus_tree"] = merged["r2_cnn"] - merged["r2_tree"]

    summary = {
        "tree_run": str(args.tree_run),
        "cnn_run": str(args.cnn_run),
        "comparison": merged.to_dict(orient="records"),
        "mean_absolute_error_change": float(merged["mae_delta_cnn_minus_tree"].mean()),
        "mean_mape_change": float(merged["mape_delta_cnn_minus_tree"].mean()),
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
