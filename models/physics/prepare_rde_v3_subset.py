"""Create a compact, stratified RDE v3 subset from the supplied ZIP archive.

The source archive is never modified.  Output waveforms remain normalized as
provided by the generator; the manifest preserves source location columns for
traceability and rewrites ``shard_file``/``row_in_shard`` for local loading.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


SPLIT_COUNTS = {
    "train": 12_000,
    "val": 2_000,
    "test_id": 1_000,
    "test_geometry": 1_000,
    "test_hard": 1_000,
    "qc_invalid": 1_000,
}
RPM_BINS = list(range(400, 3101, 100))


def archive_manifest_name(archive: ZipFile, split: str) -> str:
    suffix = f"formal/{split}/manifest.csv"
    return next(name for name in archive.namelist() if name.endswith(suffix))


def select_stratified(frame: pd.DataFrame, requested: int, seed: int, split: str) -> pd.DataFrame:
    if requested > len(frame):
        raise ValueError(f"Requested {requested} records from {split}, which has only {len(frame)}")
    working = frame.copy()
    working["_rpm_bin"] = pd.cut(working["rpm"], bins=RPM_BINS, include_lowest=True)
    columns = ["invalid_type", "_rpm_bin"] if split == "qc_invalid" else [
        "severity", "gamma_deg", "phi_deg", "d_mm", "_rpm_bin"
    ]
    groups = list(working.groupby(columns, observed=True, sort=False))
    rng = np.random.default_rng(seed)
    sizes = np.array([len(group) for _, group in groups], dtype=np.int64)
    if requested >= len(groups):
        allocation = np.ones(len(groups), dtype=np.int64)
        remaining = requested - len(groups)
        probabilities = (sizes - 1) / np.maximum((sizes - 1).sum(), 1)
        allocation += rng.multinomial(remaining, probabilities)
    else:
        allocation = rng.multinomial(requested, sizes / sizes.sum())
    selected = []
    for (_, group), count in zip(groups, allocation, strict=True):
        if count:
            selected.append(group.sample(n=min(int(count), len(group)), random_state=int(rng.integers(2**31 - 1))))
    result = pd.concat(selected, ignore_index=True).drop(columns="_rpm_bin")
    # Multinomial allocation can request more records than a small group owns.
    if len(result) < requested:
        remaining = working.loc[~working["record_index"].isin(result["record_index"])]
        result = pd.concat(
            [result, remaining.sample(n=requested - len(result), random_state=int(rng.integers(2**31 - 1)))],
            ignore_index=True,
        )
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_split(
    archive: ZipFile, prefix: str, split: str, selected: pd.DataFrame, output: Path, shard_size: int
) -> dict[str, int]:
    split_output = output / split
    split_output.mkdir(parents=True, exist_ok=True)
    source_name = selected["shard_file"].copy()
    source_row = selected["row_in_shard"].copy()
    selected = selected.copy()
    selected["source_shard_file"] = source_name
    selected["source_row_in_shard"] = source_row
    selected["shard_file"] = ""
    selected["row_in_shard"] = -1
    selected["subset_record_index"] = np.arange(len(selected), dtype=np.int64)

    buffers: list[np.ndarray] = []
    buffer_indices: list[int] = []
    current_shard = 0

    def flush() -> None:
        nonlocal buffers, buffer_indices, current_shard
        if not buffers:
            return
        filename = f"{split}_waveforms_{current_shard:04d}.npy"
        np.save(split_output / filename, np.stack(buffers).astype(np.float32, copy=False))
        selected.loc[buffer_indices, "shard_file"] = filename
        selected.loc[buffer_indices, "row_in_shard"] = np.arange(len(buffers), dtype=np.int64)
        buffers = []
        buffer_indices = []
        current_shard += 1

    ordered = selected.sort_values(["source_shard_file", "source_row_in_shard"])
    for source_file, rows in ordered.groupby("source_shard_file", sort=False):
        member = f"{prefix}/formal/{split}/{source_file}"
        source_shard = np.load(BytesIO(archive.read(member)), allow_pickle=False)
        for original_index, row in rows.iterrows():
            buffers.append(source_shard[int(row["source_row_in_shard"])])
            buffer_indices.append(int(original_index))
            if len(buffers) == shard_size:
                flush()
    flush()
    selected = selected.sort_values("subset_record_index").drop(columns="subset_record_index")
    selected.to_csv(split_output / "manifest.csv", index=False, encoding="utf-8")
    return {"records": int(len(selected)), "output_shards": current_shard}


def main() -> None:
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", type=Path, default=next(path for path in parent.glob("*/**/*.zip") if path.is_file())
    )
    parser.add_argument("--output-dir", type=Path, default=parent / "data" / "rde_v3_stratified_18000")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--shard-size", type=int, default=250)
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"source_archive": str(args.archive), "seed": args.seed, "splits": {}}
    with ZipFile(args.archive) as archive:
        prefix = archive_manifest_name(archive, "train").removesuffix("formal/train/manifest.csv").rstrip("/")
        for offset, (split, count) in enumerate(SPLIT_COUNTS.items()):
            manifest = pd.read_csv(BytesIO(archive.read(archive_manifest_name(archive, split))))
            selected = select_stratified(manifest, count, args.seed + offset, split)
            summary["splits"][split] = write_split(archive, prefix, split, selected, output, args.shard_size)
            print(f"Wrote {split}: {len(selected)} records")
        config_name = next(name for name in archive.namelist() if name.endswith("formal/config.json"))
        (output / "source_config.json").write_bytes(archive.read(config_name))
    (output / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
