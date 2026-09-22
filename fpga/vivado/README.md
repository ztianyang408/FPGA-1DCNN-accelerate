# Vivado Hardware

Target device: `xc7z020clg400-1`.

Files in this directory:

- `cnn1d.bd`: Block Design source
- `board_io.xdc`: board LED, button, switch and peripheral constraints

The generated XSA and bitstream are stored in `../release/`. The XSA is the preferred input for Vitis platform creation. The bitstream is the matching FPGA configuration image.

Vivado stores local paths in project metadata, so the original `.xpr` file is intentionally not included. Create a new project for `xc7z020clg400-1`, add `cnn1d.bd`, add `board_io.xdc`, register the HLS IP repository from `../hls_ip/phase2_rpm_cnnv3`, and regenerate the wrapper and bitstream when a rebuild is needed.
