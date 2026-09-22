from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def quantize_symmetric(arr: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float]:
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(np.max(np.abs(arr)))
    scale = max_abs / qmax if max_abs > 0 else 1.0
    quant = np.clip(np.round(arr / scale), -qmax - 1, qmax).astype(np.int8 if bits == 8 else np.int16)
    return quant, scale


def c_array_1d(name: str, values: np.ndarray, ctype: str = "float", per_line: int = 8) -> str:
    flat = values.reshape(-1)
    items = []
    for i, v in enumerate(flat):
        if ctype.startswith("int"):
            items.append(str(int(v)))
        else:
            items.append(f"{float(v):.8f}f")
    lines = [", ".join(items[i : i + per_line]) for i in range(0, len(items), per_line)]
    body = ",\n    ".join(lines)
    return f"static const {ctype} {name}[{len(flat)}] = {{\n    {body}\n}};\n"


def c_array_nd(name: str, values: np.ndarray, ctype: str = "float") -> str:
    flat = values.reshape(-1)
    shape = values.shape
    items = []
    for v in flat:
        if ctype.startswith("int"):
            items.append(str(int(v)))
        else:
            items.append(f"{float(v):.8f}f")
    body = ", ".join(items)
    return f"static const {ctype} {name}[{len(flat)}] = {{{body}}};\n// shape: {shape}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HLS assets for Phase 2")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--export-dir", type=Path, default=repo_root / "phase2" / "export")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "phase2" / "hls")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = np.load(args.export_dir / "fpga_friendly_cnn_weights.npz")
    spec = json.loads((args.export_dir / "fpga_friendly_cnn_spec.json").read_text(encoding="utf-8"))

    # Float reference headers.
    float_header = []
    float_header.append("#pragma once\n")
    float_header.append(f"#define PHASE2_INPUT_LEN {spec['input_len']}\n")
    float_header.append(f"#define PHASE2_TARGET_MEAN {float(bundle['y_mean']):.8f}f\n")
    float_header.append(f"#define PHASE2_TARGET_STD {float(bundle['y_std']):.8f}f\n")
    float_header.append(c_array_1d("phase2_train_mean", bundle["train_mean"], "float"))
    float_header.append(c_array_1d("phase2_train_std", bundle["train_std"], "float"))
    float_header.append(c_array_nd("phase2_conv1_weight", bundle["conv1_weight"], "float"))
    float_header.append(c_array_1d("phase2_conv1_bias", bundle["conv1_bias"], "float"))
    float_header.append(c_array_nd("phase2_conv2_weight", bundle["conv2_weight"], "float"))
    float_header.append(c_array_1d("phase2_conv2_bias", bundle["conv2_bias"], "float"))
    float_header.append(c_array_nd("phase2_conv3_weight", bundle["conv3_weight"], "float"))
    float_header.append(c_array_1d("phase2_conv3_bias", bundle["conv3_bias"], "float"))
    float_header.append(c_array_nd("phase2_fc1_weight", bundle["fc1_weight"], "float"))
    float_header.append(c_array_1d("phase2_fc1_bias", bundle["fc1_bias"], "float"))
    float_header.append(c_array_nd("phase2_fc2_weight", bundle["fc2_weight"], "float"))
    float_header.append(c_array_1d("phase2_fc2_bias", bundle["fc2_bias"], "float"))
    (args.out_dir / "phase2_weights_float.h").write_text("\n".join(float_header), encoding="utf-8")

    # Quantized headers for HLS.
    q_header = []
    q_header.append("#pragma once\n")
    q_header.append("#include <cstdint>\n")
    q_header.append(f"#define PHASE2_INPUT_LEN {spec['input_len']}\n")
    q_header.append(f"#define PHASE2_TARGET_MEAN {float(bundle['y_mean']):.8f}f\n")
    q_header.append(f"#define PHASE2_TARGET_STD {float(bundle['y_std']):.8f}f\n")
    q_header.append("// Symmetric per-tensor quantization; scales are stored next to each tensor.\n")

    for key in ["conv1_weight", "conv1_bias", "conv2_weight", "conv2_bias", "conv3_weight", "conv3_bias", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"]:
        quant, scale = quantize_symmetric(bundle[key], bits=8)
        q_header.append(f"#define {key.upper()}_SCALE {scale:.10f}f\n")
        q_header.append(c_array_nd(f"{key}_q", quant, "int8_t"))
        q_header.append("\n")
    q_header.append(c_array_1d("phase2_train_mean", bundle["train_mean"], "float"))
    q_header.append(c_array_1d("phase2_train_std", bundle["train_std"], "float"))
    (args.out_dir / "phase2_weights_int8.h").write_text("\n".join(q_header), encoding="utf-8")

    top_cpp = """#include "phase2_weights_int8.h"
#include <ap_fixed.h>

// Phase 2 HLS top function skeleton.
// Suggested next step: implement layer-by-layer int8 or fixed-point inference here.
// Keep the interface stable so Vivado can wrap it with AXI4-Stream or AXI4-Lite.

extern "C" {
void phase2_rpm_cnn(const float input[PHASE2_INPUT_LEN], float *output_rpm) {
    // TODO:
    // 1. Normalize input using phase2_train_mean / phase2_train_std
    // 2. Run Conv1D -> ReLU -> Pool -> Conv1D -> ReLU -> Pool -> Conv1D -> ReLU
    // 3. Global average pool
    // 4. FC32 -> FC16 -> RPM
    // 5. Dequantize output if using fixed-point internal arithmetic
    *output_rpm = 0.0f;
}
}
"""
    (args.out_dir / "phase2_rpm_cnn_top.cpp").write_text(top_cpp, encoding="utf-8")

    tb_cpp = """#include "phase2_weights_float.h"
#include <cstdio>

extern "C" void phase2_rpm_cnn(const float input[PHASE2_INPUT_LEN], float *output_rpm);

int main() {
    float input[PHASE2_INPUT_LEN] = {0};
    float output = 0.0f;
    phase2_rpm_cnn(input, &output);
    std::printf("output_rpm=%f\\n", output);
    return 0;
}
"""
    (args.out_dir / "phase2_rpm_cnn_tb.cpp").write_text(tb_cpp, encoding="utf-8")

    meta = {
        "export_dir": str(args.export_dir),
        "input_len": spec["input_len"],
        "model_type": spec["model_type"],
        "weights": {
            "float_header": "phase2_weights_float.h",
            "int8_header": "phase2_weights_int8.h",
        },
        "hls_entry": "phase2_rpm_cnn",
        "notes": [
            "Vivado cannot consume PyTorch checkpoints directly.",
            "Use int8 header for synthesis experiments and float header for reference simulation.",
            "This bundle is a bridge, not a final accelerator.",
        ],
    }
    (args.out_dir / "phase2_hls_bundle.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote HLS assets to {args.out_dir}")


if __name__ == "__main__":
    main()
