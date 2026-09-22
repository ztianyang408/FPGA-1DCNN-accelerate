# Archive Manifest

## Included

- Source training and preprocessing scripts
- Phase 1 and Phase 2 model checkpoints and evaluation summaries
- Exported FPGA weights and HLS sources
- Packaged HLS IP source
- Vivado Block Design and board constraints
- Final matching XSA and bitstream
- Vitis board application source and deterministic test vector
- Processed FFT dataset
- Technical documentation and physics-model scripts

## Excluded

- Original approximately 1.38 GB time-domain dataset
- Vivado synthesis, implementation, simulation and cache directories
- Vitis workspace, BSP and Debug output
- Generated RTL duplicated inside Vivado project output
- Historical XSA/bitstream versions
- Local machine paths and editor metadata where practical

## Final hardware pair

The files under `fpga/release/` are intended to be used together:

```text
cnn1d_wrapperv4oled1.8v.xsa
cnn1d_wrapperv4oled1.8v.bit
```

They correspond to the OLED-enabled Zynq-7020 design used by the board demonstration.
