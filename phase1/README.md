Phase 1

Goal:
- Train a baseline model on the FFT features derived from the simulated waveform dataset.
- Save metrics, predictions, plots, and the fitted model bundle here.

Suggested workflow:
1. Prepare FFT features:
   python scripts/prepare_rde_dataset.py
2. Train the baseline:
   python phase1/train_phase1_baseline.py
3. Train the lightweight CNN:
   python phase1/train_phase1_cnn.py

Outputs are written under phase1/results/.
