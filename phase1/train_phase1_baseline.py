from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "processed"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "phase1" / "results"


@dataclass
class SplitMetrics:
    split: str
    mae: float
    rmse: float
    mape: float
    r2: float


def load_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    split_dir = data_root / split
    x = np.load(split_dir / "features_fft.npy")
    y = np.load(split_dir / "targets_rpm.npy")
    meta = pd.read_csv(split_dir / "meta.csv")
    return x, y, meta


def build_model(kind: str, random_state: int):
    if kind == "tree":
        regressor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=20,
                        min_samples_leaf=5,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif kind == "rf":
        regressor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=40,
                        max_depth=18,
                        min_samples_leaf=3,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported model kind: {kind}")
    return regressor


def evaluate_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> SplitMetrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return SplitMetrics(split=name, mae=mae, rmse=rmse, mape=mape, r2=r2)


def save_scatter(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y_true, y_pred, s=8, alpha=0.45)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("True RPM")
    ax.set_ylabel("Predicted RPM")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_residual_hist(path: Path, residuals: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(residuals, bins=40, alpha=0.85, color="#4472c4")
    ax.set_title(title)
    ax.set_xlabel("Prediction Error (RPM)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def predict_in_batches(model, x: np.ndarray, batch_size: int = 1000, split_name: str | None = None) -> np.ndarray:
    outputs = []
    total = len(x)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if split_name is not None:
            print(f"  {split_name}: predicting {start}:{end} / {total}", flush=True)
        outputs.append(model.predict(x[start:end]))
    return np.concatenate(outputs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phase-1 baseline on FFT features")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--model", choices=["tree", "rf"], default="tree")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_root / f"{timestamp}_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_meta = load_split(args.data_root, "train")
    x_val, y_val, val_meta = load_split(args.data_root, "val")
    x_test_id, y_test_id, test_id_meta = load_split(args.data_root, "test_id")
    x_test_geo, y_test_geo, test_geo_meta = load_split(args.data_root, "test_geometry")
    x_test_hard, y_test_hard, test_hard_meta = load_split(args.data_root, "test_hard")

    model = build_model(args.model, args.random_state)
    print(f"Loaded data. train={x_train.shape}, val={x_val.shape}, test_id={x_test_id.shape}", flush=True)
    print(f"Fitting {args.model} baseline...", flush=True)
    model.fit(x_train, y_train)
    print("Fit complete.", flush=True)

    metrics = []
    split_specs = [
        ("train", y_train, x_train, train_meta),
        ("val", y_val, x_val, val_meta),
        ("test_id", y_test_id, x_test_id, test_id_meta),
        ("test_geometry", y_test_geo, x_test_geo, test_geo_meta),
        ("test_hard", y_test_hard, x_test_hard, test_hard_meta),
    ]
    for split_name, y_true, x_split, meta in split_specs:
        print(f"Evaluating {split_name}...", flush=True)
        y_pred = predict_in_batches(model, x_split, batch_size=1000, split_name=split_name)
        split_metrics = evaluate_split(split_name, y_true, y_pred)
        metrics.append(asdict(split_metrics))
        print(f"  {split_name}: writing predictions CSV", flush=True)
        pd.DataFrame(
            {
                "rpm_true": y_true,
                "rpm_pred": y_pred,
                "error": y_pred - y_true,
            }
        ).join(meta.reset_index(drop=True)).to_csv(run_dir / f"predictions_{split_name}.csv", index=False)
        print(f"Done {split_name}: MAE={split_metrics.mae:.3f}, RMSE={split_metrics.rmse:.3f}, MAPE={split_metrics.mape:.2f}%", flush=True)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saving model bundle...", flush=True)
    joblib.dump(model, run_dir / "model.joblib")
    summary = {
        "model": args.model,
        "data_root": str(args.data_root),
        "results_root": str(args.results_root),
        "run_dir": str(run_dir),
        "metrics": metrics,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
