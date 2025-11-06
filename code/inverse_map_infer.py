#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inverse-map inference & (optional) evaluation helper — upload‑ready.

python infer.py \
  --weights model.pth \
  --input   input.csv \
  --out     out \
  --in-dim 2003 --out-dim 2 --widths 1024,512,256,64,32,8 \
  --copy-last 4 --prepend-first 2001 \
  --clip 0:-3.141592653589793,3.141592653589793 --clip 1:0.0001,1.0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Model (must match the training architecture)
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, widths: List[int], out_dim: int, act: str = "relu", layernorm: bool = False) -> None:
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        if act not in acts:
            raise ValueError(f"Unsupported activation '{act}'. Choose from {list(acts)}")
        Act = acts[act]

        layers: List[nn.Module] = []
        prev = in_dim
        for w in widths:
            layers += [nn.Linear(prev, w)]
            if layernorm:
                layers += [nn.LayerNorm(w)]
            layers += [Act()]
            prev = w
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def auto_read(path: Path) -> np.ndarray:
    """Read numeric table (CSV/TXT). pandas auto‑detects delimiter."""
    df = pd.read_csv(path, header=None, sep=None, engine="python", dtype=np.float32)
    arr = df.values
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Invalid table shape {arr.shape} from {path}")
    return arr


# -----------------------------------------------------------------------------
# Inference routine
# -----------------------------------------------------------------------------

def run_file(
    model: nn.Module,
    device: torch.device,
    infile: Path,
    out_dir: Path,
    in_dim: int,
    out_dim: int,
    copy_last: int = 0,
    prepend_first: int = 0,
    clip_cfg: Optional[Dict[int, Tuple[float, float]]] = None,
    eval_if_targets: bool = False,
) -> None:
    arr = auto_read(infile)
    if arr.shape[1] < in_dim:
        raise ValueError(f"{infile}: columns={arr.shape[1]} < in_dim={in_dim}")

    X = arr[:, :in_dim].astype(np.float32)
    has_targets = eval_if_targets and (arr.shape[1] >= in_dim + out_dim)
    Y = arr[:, -out_dim:].astype(np.float32) if has_targets else None

    # Predict in chunks
    x_t = torch.from_numpy(X).to(device)
    model.eval()
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(x_t), 32768):
            preds.append(model(x_t[i : i + 32768]).cpu())
    pred = torch.cat(preds, dim=0).numpy()

    # Optional clipping per output dimension
    if clip_cfg:
        for j, (lo, hi) in clip_cfg.items():
            if 0 <= j < out_dim:
                pred[:, j] = np.clip(pred[:, j], lo, hi)

    # Compose output: [ copied_meta | predictions ]
    parts: List[np.ndarray] = []
    if copy_last > 0:
        if copy_last > arr.shape[1]:
            raise ValueError(f"copy_last={copy_last} exceeds input columns {arr.shape[1]}")
        parts.append(arr[:, -copy_last:])
    parts.append(pred)
    out_mat = np.hstack(parts) if len(parts) > 1 else pred

    # Optional: prepend first row (first `prepend_first` numbers from the file)
    if prepend_first > 0:
        if prepend_first > arr.shape[1]:
            raise ValueError(
                f"prepend_first={prepend_first} exceeds input columns {arr.shape[1]} in {infile}"
            )
        head_row = arr[:1, :prepend_first]
        # pad/truncate to match output width
        if head_row.shape[1] != out_mat.shape[1]:
            # If sizes differ, pad with zeros or trim to fit
            if head_row.shape[1] < out_mat.shape[1]:
                pad = np.zeros((1, out_mat.shape[1] - head_row.shape[1]), dtype=head_row.dtype)
                head_row = np.hstack([head_row, pad])
            else:
                head_row = head_row[:, : out_mat.shape[1]]
        out_mat = np.vstack([head_row, out_mat])

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{infile.stem}.csv"
    np.savetxt(out_path.as_posix(), out_mat, delimiter=",", fmt="%.10f")

    # Optional evaluation if targets present
    if has_targets:
        mse = float(np.mean((pred - Y) ** 2))
        mae = float(np.mean(np.abs(pred - Y)))
        metrics = {"mse": mse, "mae": mae, "n_samples": int(X.shape[0])}
        with (out_dir / f"metrics_{infile.stem}.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[EVAL] {infile.name}  MSE={mse:.6e}  MAE={mae:.6e}  N={X.shape[0]}")
    else:
        print(f"[SAVE] {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_clip(items: List[str], out_dim: int) -> Dict[int, Tuple[float, float]]:
    """Parse --clip like: 0:-3.14,3.14  1:0.0001,1.0  (index:min,max)."""
    res: Dict[int, Tuple[float, float]] = {}
    for it in items:
        try:
            idx_str, rng = it.split(":", 1)
            lo_str, hi_str = rng.split(",", 1)
            idx = int(idx_str)
            lo = float(lo_str)
            hi = float(hi_str)
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid --clip '{it}', expected like 0:-3.14,3.14") from e
        if not (0 <= idx < out_dim):
            raise argparse.ArgumentTypeError(f"clip index {idx} not in [0,{out_dim})")
        res[idx] = (lo, hi)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Inverse-map inference & optional evaluation")
    ap.add_argument("--weights", type=Path, required=True, help="Path to checkpoint .pth (state_dict or dict)")
    ap.add_argument("--input", type=Path, nargs="+", required=True, help="One or more input CSV/TXT files")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for predictions/metrics")

    # Model hyper-params (must match training)
    ap.add_argument("--in-dim", type=int, default=2003)
    ap.add_argument("--out-dim", type=int, default=2)
    ap.add_argument("--widths", type=str, default="1024,512,256,64,32,8")
    ap.add_argument("--act", type=str, default="relu", choices=["relu", "gelu", "tanh", "silu"])
    ap.add_argument("--layernorm", action="store_true", help="Enable LayerNorm between layers (if used in training)")

    # Output formatting options
    ap.add_argument("--copy-last", type=int, default=0, help="Copy the last N columns from input to the output prefix")
    ap.add_argument(
        "--prepend-first",
        type=int,
        default=0,
        help="Prepend a single row taken from the input's first row (first M numbers)",
    )
    ap.add_argument(
        "--clip",
        type=str,
        nargs="*",
        default=[],
        help="Per-dimension clipping like: 0:-3.14159,3.14159 1:0.0001,1.0",
    )

    # Misc
    ap.add_argument("--eval-if-targets", action="store_true", help="If input has targets at the end, compute MSE/MAE")
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Override auto device")

    args = ap.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Build model & load weights
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    model = MLP(args.in_dim, widths, args.out_dim, act=args.act, layernorm=args.layernorm).to(device)

    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Parse clipping config
    clip_cfg = parse_clip(args.clip, args.out_dim) if args.clip else None

    # Save a run manifest
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "infer_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": str(args.weights),
                "inputs": [str(p) for p in args.input],
                "in_dim": args.in_dim,
                "out_dim": args.out_dim,
                "widths": widths,
                "act": args.act,
                "layernorm": bool(args.layernorm),
                "copy_last": int(args.copy_last),
                "prepend_first": int(args.prepend_first),
                "clip": clip_cfg,
                "device": str(device),
                "eval_if_targets": bool(args.eval_if_targets),
            },
            f,
            indent=2,
        )

    # Process each input file
    for infile in args.input:
        run_file(
            model=model,
            device=device,
            infile=infile,
            out_dir=args.out,
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            copy_last=args.copy_last,
            prepend_first=args.prepend_first,
            clip_cfg=clip_cfg,
            eval_if_targets=args.eval_if_targets,
        )


if __name__ == "__main__":
    main()
