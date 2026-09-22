Phase 2

Goal:
- Build an FPGA-friendly CNN variant from the Phase 1 pipeline.
- Keep the same 512-dim FFT input and RPM regression target.
- Reduce channel width, remove BatchNorm/Dropout/GELU, and use ReLU + fixed pooling.

Workflow:
1. Prepare FFT features:
   python scripts/prepare_rde_dataset.py
2. Train the FPGA-friendly CNN:
   python phase2/train_phase2_fpga_cnn.py
3. Export FPGA artifacts:
   python phase2/export_phase2_fpga_bundle.py
4. Prepare HLS assets:
   python phase2/prepare_hls_assets.py
5. Open Vitis HLS, run C Simulation and Synthesis on:
   phase2/hls/phase2_rpm_cnn_top.cpp
   phase2/hls/phase2_rpm_cnn_tb.cpp
6. Export the HLS IP, then integrate it in Vivado.
7. Build the ARM-side host app in Vitis and test on board.

Outputs are written under phase2/results/.

HLS implementation notes:
- The first HLS reference version used float arithmetic and exceeded xc7z020 resources.
- The current top implementation uses saturated ap_fixed activations/accumulators and int8 exported weights.
- The current synthesis-tuning pass uses II=8 for convolution loops and II=16 for dense loops to reduce DSP pressure.
- Quantized layers now accumulate input * int8_weight first, then apply the per-layer scale once per output element.
- Accumulators must keep enough integer range; narrow accumulators can wrap and produce plausible but wrong board outputs.
- After synthesis, check DSP/LUT/BRAM utilization first, then latency.
- If resources still exceed xc7z020, reduce channel width or lower data_t/acc_t bit widths.
