# Data

`processed/` contains the FFT feature representation used by Phase 1 and Phase 2.

Each split contains:

- `features_fft.npy`: shape `(N, 512)`, log-magnitude FFT features
- `targets_rpm.npy`: RPM regression labels
- `meta.csv`: geometry and sample metadata
- `feature_config.json`: sampling and frequency-band configuration

Current preprocessing configuration:

- sample rate: 20 kHz
- frequency band: 20 to 1600 Hz
- feature length: 512
- transform: `log1p(abs(rFFT(signal)))`

The original time-domain simulation data is not included in the repository because it is large. To regenerate processed data, obtain the original dataset separately and run:

```bash
python scripts/prepare_rde_dataset.py \
  --data-root path/to/rde_v3_stratified_18000 \
  --out-root data/processed
```

`test_vector/` contains the deterministic board-level test vector and its expected values.
