from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE1_RUN = REPO_ROOT / "phase1" / "results"
DEFAULT_PHASE2_RUN = REPO_ROOT / "phase2" / "results"


def load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return pd.read_csv(metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 2 against Phase 1")
    parser.add_argument("--phase1-run", type=Path, default=DEFAULT_PHASE1_RUN)
    parser.add_argument("--phase2-run", type=Path, default=DEFAULT_PHASE2_RUN)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "phase2" / "results" / "phase2_vs_phase1.json")
    args = parser.parse_args()

    phase1 = load_metrics(args.phase1_run)
    phase2 = load_metrics(args.phase2_run)
    merged = phase1.merge(phase2, on="split", suffixes=("_phase1", "_phase2"))
    merged["mae_delta_phase2_minus_phase1"] = merged["mae_phase2"] - merged["mae_phase1"]
    merged["rmse_delta_phase2_minus_phase1"] = merged["rmse_phase2"] - merged["rmse_phase1"]
    merged["mape_delta_phase2_minus_phase1"] = merged["mape_phase2"] - merged["mape_phase1"]
    merged["r2_delta_phase2_minus_phase1"] = merged["r2_phase2"] - merged["r2_phase1"]

    summary = {
        "phase1_run": str(args.phase1_run),
        "phase2_run": str(args.phase2_run),
        "comparison": merged.to_dict(orient="records"),
        "mean_mae_change": float(merged["mae_delta_phase2_minus_phase1"].mean()),
        "mean_mape_change": float(merged["mape_delta_phase2_minus_phase1"].mean()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
