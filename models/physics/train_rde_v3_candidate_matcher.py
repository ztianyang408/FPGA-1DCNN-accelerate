"""Range-equivariant harmonic candidate matcher for RPM extrapolation.

Instead of regressing RPM, the model scores every candidate rotation frequency
with the same MLP. Candidate features are PSD samples at m*f_rot and their local
contrast. Shared scoring weights cannot encode a preferred training RPM range.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_rde_v3_spectral_cnn import SPLITS, set_seed, spectral_features


ORDERS = 32
CANDIDATE_STEP_RPM = 10.0


class CandidateMatcher(nn.Module):
    def __init__(self, candidates_rpm: torch.Tensor, frequency_hz: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("candidates_rpm", candidates_rpm)
        query = candidates_rpm[:, None] / 60.0 * torch.arange(
            1, ORDERS + 1, device=candidates_rpm.device, dtype=candidates_rpm.dtype
        )[None]
        # Normalized grid coordinate for linear grid_sample on the PSD axis.
        grid = 2.0 * (query - frequency_hz[0]) / (frequency_hz[-1] - frequency_hz[0]) - 1.0
        self.register_buffer("query_grid", grid)
        self.register_buffer("left_grid", grid - 2.0 * 0.75 / (frequency_hz[-1] - frequency_hz[0]))
        self.register_buffer("right_grid", grid + 2.0 * 0.75 / (frequency_hz[-1] - frequency_hz[0]))
        valid = (query >= frequency_hz[0]) & (query <= frequency_hz[-1])
        self.register_buffer("valid", valid.float())
        # Shared over all candidate RPM values. No candidate RPM itself enters.
        self.order_encoder = nn.Sequential(
            nn.Linear(4, 24), nn.GELU(), nn.Linear(24, 16), nn.GELU()
        )
        self.score_head = nn.Sequential(
            nn.Linear(16, 32), nn.GELU(), nn.Dropout(0.10), nn.Linear(32, 1)
        )
        # A shallow translation-equivariant filter suppresses broadband noise.
        self.denoiser = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.GELU(),
            nn.Conv1d(16, 16, 7, padding=3), nn.GELU(),
            nn.Conv1d(16, 1, 5, padding=2),
        )

    @staticmethod
    def _sample(psd: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        batch = len(psd)
        # Treat 1-D spectrum as a height-one image for bilinear interpolation.
        image = psd[:, None, None, :]
        x = grid[None].expand(batch, -1, -1)
        y = torch.zeros_like(x)
        sample_grid = torch.stack([x, y], dim=-1)
        return F.grid_sample(image, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]

    def forward(self, psd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        filtered = psd + 0.25 * self.denoiser(psd[:, None]).squeeze(1)
        centre = self._sample(filtered, self.query_grid)
        left = self._sample(filtered, self.left_grid)
        right = self._sample(filtered, self.right_grid)
        contrast = centre - 0.5 * (left + right)
        order_position = torch.linspace(0.0, 1.0, ORDERS, device=psd.device)[None, None].expand_as(centre)
        valid = self.valid[None].expand_as(centre)
        features = torch.stack([centre, contrast, order_position, valid], dim=-1)
        encoded = self.order_encoder(features)
        # Masked mean gives every candidate the same-scale representation.
        pooled = torch.sum(encoded * valid[..., None], dim=2) / torch.clamp(valid.sum(dim=2, keepdim=True), min=1.0)
        scores = self.score_head(pooled).squeeze(-1)
        probabilities = torch.softmax(scores, dim=1)
        prediction = probabilities @ self.candidates_rpm
        return scores, prediction


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    total = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "mape_pct": float(np.mean(np.abs(error) / actual) * 100),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
        "r2": float(1.0 - np.sum(error**2) / max(total, 1e-12)),
    }


def partition(raw, minimum: float, maximum: float):
    result = {}
    def add(source, name, lower, upper):
        shape, _, frame = raw[source]
        mask = np.ones(len(frame), bool)
        if lower is not None: mask &= frame.rpm.to_numpy() >= lower
        if upper is not None: mask &= frame.rpm.to_numpy() <= upper
        result[name] = (shape[mask], frame.loc[mask].reset_index(drop=True))
    add("train", "train_core", minimum, maximum)
    add("val", "val_core", minimum, maximum)
    for source in ("test_id", "test_geometry", "test_hard"):
        add(source, f"{source}_core", minimum, maximum)
        add(source, f"{source}_ood_low", None, minimum - 1e-6)
        add(source, f"{source}_ood_high", maximum + 1e-6, None)
    return result


def predict(model, shape, device):
    model.eval(); output=[]
    with torch.no_grad():
        for start in range(0,len(shape),128):
            _, rpm=model(torch.from_numpy(shape[start:start+128]).to(device)); output.append(rpm.cpu().numpy())
    return np.concatenate(output)


def main():
    parent=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,default=parent/"data"/"rde_v3_stratified_18000")
    parser.add_argument("--output-dir",type=Path,default=parent/"results"/"rde_v3_candidate_matcher")
    parser.add_argument("--epochs",type=int,default=90)
    parser.add_argument("--seed",type=int,default=20260810)
    args=parser.parse_args(); set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw={s:spectral_features(args.data_root,s,20,1600) for s in SPLITS}
    data=partition(raw,700,2400)
    frequency=torch.arange(20.0,1600.01,1.0,device=device)
    candidates=torch.arange(400.0,3000.01,CANDIDATE_STEP_RPM,device=device)
    model=CandidateMatcher(candidates,frequency).to(device)
    train_x,train_frame=data["train_core"]
    loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_frame.rpm.to_numpy(np.float32))),batch_size=64,shuffle=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    best=None; best_mae=float("inf"); patience=0; history=[]
    for epoch in range(args.epochs):
        model.train(); total=0; count=0
        for psd,rpm in loader:
            psd=psd.to(device)+0.01*torch.randn_like(psd.to(device)); rpm=rpm.to(device)
            scores,pred=model(psd)
            # Gaussian soft target prevents grid quantization and preserves a smooth score surface.
            target=torch.exp(-0.5*((model.candidates_rpm[None]-rpm[:,None])/20.0)**2)
            target=target/target.sum(dim=1,keepdim=True)
            classification=-(target*torch.log_softmax(scores,dim=1)).sum(dim=1).mean()
            regression=F.smooth_l1_loss(pred/1000.0,rpm/1000.0)
            loss=classification+0.5*regression
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),3); optimizer.step()
            total+=float(loss.detach())*len(psd); count+=len(psd)
        scheduler.step()
        val_x,val_frame=data["val_core"]; val_pred=predict(model,val_x,device)
        val=metrics(val_frame.rpm.to_numpy(np.float32),val_pred)
        history.append({"epoch":epoch+1,"train_loss":total/count,**{f"val_{k}":v for k,v in val.items()}})
        if val["mae_rpm"]<best_mae: best_mae=val["mae_rpm"];best=copy.deepcopy(model.state_dict());patience=0
        else: patience+=1
        if (epoch+1)%10==0: print(f"epoch {epoch+1}: val MAE {val['mae_rpm']:.2f}, MAPE {val['mape_pct']:.2f}%",flush=True)
        if patience>=18: break
    model.load_state_dict(best); args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":best},args.output_dir/"best_model.pt"); pd.DataFrame(history).to_csv(args.output_dir/"history.csv",index=False)
    report={"protocol":"Shared harmonic candidate scorer; train 700-2400 RPM only; candidates 400-3000 RPM.","results":{}}
    rows=[]
    for name,(shape,frame) in data.items():
        pred=predict(model,shape,device); actual=frame.rpm.to_numpy(np.float32)
        report["results"][name]={"samples":len(actual),**metrics(actual,pred)}
        out=frame[["rpm","severity","record_index"]].copy();out["split"]=name;out["prediction_rpm"]=pred;rows.append(out)
    pd.concat(rows).to_csv(args.output_dir/"predictions.csv",index=False)
    (args.output_dir/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))

if __name__=="__main__": main()
