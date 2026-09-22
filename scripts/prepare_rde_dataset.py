from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SPLITS = ["train", "val", "test_id", "test_geometry", "test_hard", "qc_invalid"]


def load_manifest(split_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(split_dir / "manifest.csv")
    manifest["shard_file"] = manifest["shard_file"].astype(str)
    manifest["row_in_shard"] = manifest["row_in_shard"].astype(int)
    return manifest


def summarize_dataset(data_root: Path) -> None:
    print("Dataset summary")
    for split in DEFAULT_SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        manifest = load_manifest(split_dir)
        valid_counts = manifest["valid"].value_counts(dropna=False).to_dict() if "valid" in manifest else {}
        print(f"- {split}: rows={len(manifest)}, valid={valid_counts}, shards={manifest['shard_file'].nunique()}")
        if "rpm" in manifest:
            print(f"  rpm range: {manifest['rpm'].min():.3f} .. {manifest['rpm'].max():.3f}")
        if "severity" in manifest:
            sev = manifest["severity"].value_counts(dropna=False).to_dict()
            print(f"  severity: {sev}")


def fft_features(signal: np.ndarray, sample_rate_hz: float, freq_low_hz: float, freq_high_hz: float, feature_len: int) -> np.ndarray:
    spectrum = np.fft.rfft(signal.astype(np.float32), axis=-1)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(signal.shape[-1], d=1.0 / sample_rate_hz)
    mask = (freqs >= freq_low_hz) & (freqs <= freq_high_hz)
    band = magnitude[..., mask]
    band = np.log1p(band).astype(np.float32)
    if band.shape[-1] == feature_len:
        return band
    x_old = np.linspace(0.0, 1.0, band.shape[-1], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, feature_len, dtype=np.float32)
    return np.interp(x_new, x_old, band).astype(np.float32)


def export_split(
    data_root: Path,
    out_root: Path,
    split: str,
    feature_len: int,
    freq_low_hz: float,
    freq_high_hz: float,
    limit: int | None = None,
) -> None:
    split_dir = data_root / split
    manifest = load_manifest(split_dir)
    if limit is not None:
        manifest = manifest.iloc[:limit].copy()

    sample_rate_hz = 20000.0
    groups = manifest.groupby("shard_file", sort=False)
    features = np.empty((len(manifest), feature_len), dtype=np.float32)
    targets = np.empty((len(manifest),), dtype=np.float32)

    cursor = 0
    for shard_file, shard_rows in groups:
        shard = np.load(split_dir / shard_file)
        row_indices = shard_rows["row_in_shard"].to_numpy(dtype=np.int64)
        waveforms = shard[row_indices]
        feats = np.stack(
            [fft_features(w, sample_rate_hz, freq_low_hz, freq_high_hz, feature_len) for w in waveforms],
            axis=0,
        )
        n = len(shard_rows)
        features[cursor : cursor + n] = feats
        targets[cursor : cursor + n] = shard_rows["rpm"].to_numpy(dtype=np.float32)
        cursor += n

    meta_cols = [
        c
        for c in [
            "rpm",
            "recovered_rpm",
            "gamma_deg",
            "phi_deg",
            "d_mm",
            "severity",
            "valid",
            "shard_file",
            "row_in_shard",
            "record_index",
            "scene_index",
            "speed_ripple_pct",
            "speed_wander_pct",
        ]
        if c in manifest.columns
    ]
    meta = manifest[meta_cols].copy()

    split_out = out_root / split
    split_out.mkdir(parents=True, exist_ok=True)
    np.save(split_out / "features_fft.npy", features)
    np.save(split_out / "targets_rpm.npy", targets)
    meta.to_csv(split_out / "meta.csv", index=False)

    config = {
        "split": split,
        "sample_rate_hz": sample_rate_hz,
        "freq_low_hz": freq_low_hz,
        "freq_high_hz": freq_high_hz,
        "feature_len": feature_len,
        "rows": len(manifest),
    }
    (split_out / "feature_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Exported {split}: {len(manifest)} rows -> {split_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FFT features from the RDE dataset")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument("--out-root", type=Path, default=repo_root / "data" / "processed")
    parser.add_argument("--feature-len", type=int, default=512)
    parser.add_argument("--freq-low", type=float, default=20.0)
    parser.add_argument("--freq-high", type=float, default=1600.0)
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test_id", "test_geometry", "test_hard"], help="Splits to export")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per split for quick smoke tests")
    parser.add_argument("--summary", action="store_true", help="Print a dataset summary and exit")
    args = parser.parse_args()

    if args.summary:
        summarize_dataset(args.data_root)
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    source_config = args.data_root / "source_config.json"
    if source_config.exists():
        (args.out_root / "source_config.json").write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    for split in args.splits:
        export_split(args.data_root, args.out_root, split, args.feature_len, args.freq_low, args.freq_high, args.limit)


if __name__ == "__main__":
    main()
