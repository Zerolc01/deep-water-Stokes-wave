#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference for the ML series predictor.

- Loads a trained checkpoint and produces predictions for an input file.

Examples
--------
# Inference on X-only file
python infer.py \
  --weights outputs/run1/checkpoints/best.pth \
  --input  data/input.txt \
  --out    outputs/infer_plot \
  --out-dim 2001 --hidden 64 --layers 12

# Evaluate on a held-out test set (X+Y)
python infer.py \
  --weights outputs/run1/checkpoints/best.pth \
  --test   data/test.txt \
  --out    outputs/infer_test \
  --out-dim 2001 --hidden 64 --layers 12

# Evaluate and also predict for an X-only file
python infer.py \
  --weights outputs/run1/checkpoints/best.pth \
  --input  data/input.txt \
  --test   data/test.txt \
  --out    outputs/infer_both \
  --out-dim 2001 --hidden 64 --layers 12
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Model (must match training architecture)
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int, act: str = "relu") -> None:
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if act not in acts:
            raise ValueError(f"Unsupported activation '{act}'. Choose from {list(acts)}")
        Act = acts[act]

        blocks = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), Act()]
        for _ in range(max(0, layers - 1)):
            blocks += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), Act()]
        blocks += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def load_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path.as_posix(), delimiter=",", ndmin=2)


def split_xy(arr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns (X, Y or None)."""
    if arr.shape[1] == 1:
        return arr[:, :1], None
    return arr[:, :1], arr[:, 1:]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference & test evaluation for ML series predictor")
    parser.add_argument("--weights", type=Path, required=True, help="Path to checkpoint .pth (state dict or dict)")
    parser.add_argument("--input", type=Path, default=None, help="Path to X-only or X+Y file for prediction")
    parser.add_argument("--test", type=Path, default=None, help="Optional X+Y test file for evaluation")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")

    # Model hyper-params (must match training!)
    parser.add_argument("--in-dim", type=int, default=1)
    parser.add_argument("--out-dim", type=int, default=2001)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--act", type=str, default="relu", choices=["relu", "gelu", "tanh", "silu"])

    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Override auto device")
    args = parser.parse_args()

    # Avoid OpenMP duplicate symbol issues in some envs
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Build model and load weights
    model = MLP(args.in_dim, args.out_dim, args.hidden, args.layers, args.act).to(device)

    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)  # support plain state_dict
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Save a small run manifest
    with (out_dir / "infer_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": str(args.weights),
                "input": str(args.input) if args.input else None,
                "test": str(args.test) if args.test else None,
                "in_dim": args.in_dim,
                "out_dim": args.out_dim,
                "hidden": args.hidden,
                "layers": args.layers,
                "act": args.act,
                "device": str(device),
            },
            f,
            indent=2,
        )

    # ---------------- Inference on --input ----------------
    if args.input is not None:
        arr = load_txt(args.input)
        X, Y_opt = split_xy(arr)
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_t), 2048):
                preds.append(model(X_t[i : i + 2048]).cpu())
            pred = torch.cat(preds, dim=0)
        np.savetxt((out_dir / "pred_input.csv").as_posix(), pred.numpy(), delimiter=",")

        # If input also contains ground truth, evaluate against it
        if Y_opt is not None:
            Y_t = torch.from_numpy(Y_opt.astype(np.float32))
            mse = torch.mean((pred - Y_t) ** 2).item()
            mae = torch.mean(torch.abs(pred - Y_t)).item()
            with (out_dir / "metrics_input.json").open("w", encoding="utf-8") as f:
                json.dump({"mse": mse, "mae": mae, "n_samples": int(X.shape[0])}, f, indent=2)
            print(f"[INPUT] MSE={mse:.6e}, MAE={mae:.6e}, N={X.shape[0]}")
        else:
            print(f"Saved predictions for input file to {out_dir / 'pred_input.csv'}")

    # ---------------- Evaluation on --test ----------------
    if args.test is not None:
        arr_t = load_txt(args.test)
        X_test, Y_test = split_xy(arr_t)
        if Y_test is None:
            raise ValueError("--test file must contain targets (>=2 columns)")
        Xv = torch.from_numpy(X_test.astype(np.float32)).to(device)
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xv), 2048):
                preds.append(model(Xv[i : i + 2048]).cpu())
            pred_t = torch.cat(preds, dim=0)
        np.savetxt((out_dir / "pred_test.csv").as_posix(), pred_t.numpy(), delimiter=",")
        Yv = torch.from_numpy(Y_test.astype(np.float32))
        mse = torch.mean((pred_t - Yv) ** 2).item()
        mae = torch.mean(torch.abs(pred_t - Yv)).item()
        with (out_dir / "metrics_test.json").open("w", encoding="utf-8") as f:
            json.dump({"mse": mse, "mae": mae, "n_samples": int(X_test.shape[0])}, f, indent=2)
        print(f"[TEST]  MSE={mse:.6e}, MAE={mae:.6e}, N={X_test.shape[0]}")


if __name__ == "__main__":
    main()
