from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class FPGAFriendlyRPMCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.mean(dim=-1)
        return self.head(x).squeeze(-1)


@dataclass
class LayerShape:
    name: str
    shape: list[int]


def latest_checkpoint(results_root: Path) -> Path:
    candidates = []
    direct = results_root / "fpga_friendly_cnn.pt"
    if direct.exists():
        candidates.append(direct)
    candidates.extend(sorted(results_root.glob("*_fpga_cnn/fpga_friendly_cnn.pt")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {results_root}")
    return candidates[-1]


def export_numpy_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value.astype(np.float32, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 2 FPGA-friendly CNN artifacts")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Phase 2 checkpoint file")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-root", type=Path, default=repo_root / "phase2" / "results")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "phase2" / "export")
    parser.add_argument("--onnx", action="store_true", help="Try to export ONNX if the package is installed")
    args = parser.parse_args()

    checkpoint = args.checkpoint or latest_checkpoint(args.results_root)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)

    model = FPGAFriendlyRPMCNN()
    model.load_state_dict(bundle["model_state"])
    model.eval()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the model in a form that is easier to compare with Python inference.
    torch.save(model.state_dict(), out_dir / "fpga_friendly_cnn_state_dict.pt")
    torch.jit.script(model).save(out_dir / "fpga_friendly_cnn_torchscript.pt")

    # Export layer weights and normalization parameters for HLS / RTL integration.
    weights = {
        "conv1_weight": model.features[0].weight.detach().cpu().numpy(),
        "conv1_bias": model.features[0].bias.detach().cpu().numpy(),
        "conv2_weight": model.features[3].weight.detach().cpu().numpy(),
        "conv2_bias": model.features[3].bias.detach().cpu().numpy(),
        "conv3_weight": model.features[6].weight.detach().cpu().numpy(),
        "conv3_bias": model.features[6].bias.detach().cpu().numpy(),
        "fc1_weight": model.head[0].weight.detach().cpu().numpy(),
        "fc1_bias": model.head[0].bias.detach().cpu().numpy(),
        "fc2_weight": model.head[2].weight.detach().cpu().numpy(),
        "fc2_bias": model.head[2].bias.detach().cpu().numpy(),
        "train_mean": np.asarray(bundle["train_mean"], dtype=np.float32),
        "train_std": np.asarray(bundle["train_std"], dtype=np.float32),
        "y_mean": np.asarray(bundle["y_mean"], dtype=np.float32),
        "y_std": np.asarray(bundle["y_std"], dtype=np.float32),
    }
    np.savez(out_dir / "fpga_friendly_cnn_weights.npz", **weights)

    shapes = [
        LayerShape("input", [1, int(bundle["input_len"])]),
        LayerShape("conv1_weight", list(weights["conv1_weight"].shape)),
        LayerShape("conv2_weight", list(weights["conv2_weight"].shape)),
        LayerShape("conv3_weight", list(weights["conv3_weight"].shape)),
        LayerShape("fc1_weight", list(weights["fc1_weight"].shape)),
        LayerShape("fc2_weight", list(weights["fc2_weight"].shape)),
    ]
    spec = {
        "checkpoint": str(checkpoint),
        "model_type": "fpga_friendly_1d_cnn",
        "input_len": int(bundle["input_len"]),
        "input_shape": [1, int(bundle["input_len"])],
        "normalization": {
            "train_mean": "stored in fpga_friendly_cnn_weights.npz",
            "train_std": "stored in fpga_friendly_cnn_weights.npz",
            "target_mean": float(bundle["y_mean"]),
            "target_std": float(bundle["y_std"]),
        },
        "layers": [asdict(item) for item in shapes],
        "deployment_notes": [
            "Vivado does not ingest PyTorch directly.",
            "Use these exported tensors to implement or verify a Vitis HLS / RTL accelerator.",
            "Keep ReLU and pooling; avoid BatchNorm and Dropout in the hardware path.",
            "Quantize before writing the final accelerator.",
        ],
    }
    (out_dir / "fpga_friendly_cnn_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    if args.onnx:
        try:
            import onnx  # type: ignore  # noqa: F401

            dummy = torch.zeros(1, int(bundle["input_len"]), dtype=torch.float32)
            torch.onnx.export(
                model,
                dummy,
                out_dir / "fpga_friendly_cnn.onnx",
                input_names=["fft_features"],
                output_names=["rpm"],
                opset_version=17,
                dynamic_axes={"fft_features": {0: "batch"}, "rpm": {0: "batch"}},
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            (out_dir / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")

    print(f"Exported checkpoint: {checkpoint}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
