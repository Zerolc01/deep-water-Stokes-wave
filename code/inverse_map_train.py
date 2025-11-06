#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an inverse conformal mapping network: R^{in_dim} -> R^{out_dim} (default 2003 -> 2).

Expected data format (rows):
    x1, x2, ..., x_in_dim, y1, y2
where the last `out_dim` columns are targets. Default: in_dim=2003, out_dim=2.

Examples
--------
python train.py \
  --train data/train.csv \
  --test  data/test.csv \
  --out   outputs/inverse_map_run1 \
  --in-dim 2003 --out-dim 2 \
  --widths 1024,512,256,64,32,8 --epochs 2000 --batch-size 32768 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def set_deterministic(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, skip_rows: int = 0) -> np.ndarray:
    """Read numeric table with pandas (auto delimiter). Returns float32 ndarray."""
    df = pd.read_csv(
        path,
        header=None,
        skiprows=skip_rows,
        sep=None,
        engine="python",
        dtype=np.float32,
    )
    arr = df.values
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Data must be 2D with >=3 columns, got shape {arr.shape} from {path}")
    return arr


def split_xy(arr: np.ndarray, out_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(arr[:, :-out_dim])
    y = torch.from_numpy(arr[:, -out_dim:])
    return x, y


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, widths: List[int], out_dim: int, act: str = "relu", layernorm: bool = True):
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
# Training
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    train: Path
    test: Optional[Path]
    out: Path
    in_dim: int = 2003
    out_dim: int = 2
    widths: str = "1024,512,256,64,32,8"
    act: str = "relu"
    layernorm: bool = True
    epochs: int = 2000
    batch_size: int = 32768
    lr: float = 1e-3
    weight_decay: float = 1e-4
    cosine_tmax: Optional[int] = None
    patience: int = 100
    min_delta: float = 0.0
    clip_grad: float = 1.0
    amp: bool = False
    num_workers: int = 0
    device: Optional[str] = None
    seed: int = 42
    skip_rows: int = 1  # your original files had one header line


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_arr = read_table(cfg.train, cfg.skip_rows)
    if train_arr.shape[1] != cfg.in_dim + cfg.out_dim:
        raise ValueError(
            f"Train columns mismatch: got {train_arr.shape[1]}, expected {cfg.in_dim + cfg.out_dim}"
        )
    x_tr, y_tr = split_xy(train_arr, cfg.out_dim)
    ds_tr = TensorDataset(x_tr, y_tr)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=min(cfg.batch_size, len(ds_tr)),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dl_val = None
    if cfg.test is not None:
        test_arr = read_table(cfg.test, cfg.skip_rows)
        if test_arr.shape[1] != cfg.in_dim + cfg.out_dim:
            raise ValueError(
                f"Test columns mismatch: got {test_arr.shape[1]}, expected {cfg.in_dim + cfg.out_dim}"
            )
        x_te, y_te = split_xy(test_arr, cfg.out_dim)
        ds_te = TensorDataset(x_te, y_te)
        dl_val = DataLoader(
            ds_te,
            batch_size=min(cfg.batch_size, len(ds_te)),
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return dl_tr, dl_val


def train(cfg: TrainConfig) -> None:
    set_deterministic(cfg.seed)

    device = torch.device(cfg.device) if cfg.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    ensure_dir(cfg.out)
    ckpt_dir = cfg.out / "checkpoints"
    fig_dir = cfg.out / "figs"
    ensure_dir(ckpt_dir)
    ensure_dir(fig_dir)

    with (cfg.out / "config.json").open("w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}, f, indent=2)

    # Data
    dl_tr, dl_val = build_loaders(cfg)

    # Model
    widths = [int(x) for x in cfg.widths.split(",") if x.strip()]
    model = MLP(cfg.in_dim, widths, cfg.out_dim, act=cfg.act, layernorm=cfg.layernorm).to(device)

    # Optim & sched
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.cosine_tmax or cfg.epochs, eta_min=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    crit = nn.MSELoss()

    best_val = math.inf
    best_epoch = -1
    no_improve = 0

    hist = {"train_mse": [], "train_mae": [], "val_mse": [], "val_mae": [], "lr": []}

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Epochs"):
        model.train()
        mse_e = 0.0
        mae_e = 0.0
        n_batches = 0

        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                pred = model(xb)
                loss = crit(pred, yb)
            scaler.scale(loss).backward()
            if cfg.clip_grad and cfg.clip_grad > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optim)
            scaler.update()

            with torch.no_grad():
                mse_e += torch.mean((pred - yb) ** 2).item()
                mae_e += torch.mean(torch.abs(pred - yb)).item()
                n_batches += 1

        sched.step()
        mse_e /= max(1, n_batches)
        mae_e /= max(1, n_batches)
        hist["train_mse"].append(mse_e)
        hist["train_mae"].append(mae_e)
        hist["lr"].append(optim.param_groups[0]["lr"])

        # Validation
        val_mse, val_mae = float("nan"), float("nan")
        if dl_val is not None:
            model.eval()
            se_sum = 0.0
            ae_sum = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = model(xb)
                    se_sum += torch.sum((pred - yb) ** 2).item()
                    ae_sum += torch.sum(torch.abs(pred - yb)).item()
                    n += yb.numel()
            val_mse = se_sum / n
            val_mae = ae_sum / n
            hist["val_mse"].append(val_mse)
            hist["val_mae"].append(val_mae)

            improved = (best_val - val_mse) > cfg.min_delta
            if improved:
                best_val = val_mse
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optim.state_dict(),
                        "config": asdict(cfg),
                        "best_val_mse": best_val,
                    },
                    ckpt_dir / "best.pth",
                )
            else:
                no_improve += 1

            # Always save last
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "config": asdict(cfg),
                    "best_val_mse": best_val,
                },
                ckpt_dir / "last.pth",
            )

            if cfg.patience > 0 and no_improve >= cfg.patience:
                print(f"Early stop at epoch {epoch} (best @ {best_epoch}, val_mse={best_val:.6e})")
                break

        # Compact log
        if dl_val is None:
            tqdm.write(f"Epoch {epoch:04d} | train_mse={mse_e:.3e} train_mae={mae_e:.3e}")
        else:
            tqdm.write(
                f"Epoch {epoch:04d} | train_mse={mse_e:.3e} train_mae={mae_e:.3e} | "
                f"val_mse={val_mse:.3e} val_mae={val_mae:.3e}"
            )

    # Save history
    with (cfg.out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    # Curves
    try:
        plt.figure(figsize=(8,5))
        plt.plot(hist["train_mse"], label="train MSE")
        if len(hist["val_mse"]) > 0:
            plt.plot(hist["val_mse"], label="val MSE")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (log)")
        plt.legend(); plt.tight_layout()
        plt.savefig((fig_dir / "mse_curve.png").as_posix(), dpi=200); plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(hist["train_mae"], label="train MAE")
        if len(hist["val_mae"]) > 0:
            plt.plot(hist["val_mae"], label="val MAE")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MAE (log)")
        plt.legend(); plt.tight_layout()
        plt.savefig((fig_dir / "mae_curve.png").as_posix(), dpi=200); plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(hist["lr"], label="learning rate")
        plt.xlabel("Epoch"); plt.ylabel("LR"); plt.tight_layout()
        plt.savefig((fig_dir / "lr.png").as_posix(), dpi=200); plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train inverse mapping network (upload‑ready)")
    p.add_argument("--train", type=Path, required=True, help="Path to training CSV/TXT")
    p.add_argument("--test", type=Path, default=None, help="Optional validation CSV/TXT")
    p.add_argument("--out", type=Path, required=True, help="Output directory")

    p.add_argument("--in-dim", type=int, default=2003)
    p.add_argument("--out-dim", type=int, default=2)
    p.add_argument("--widths", type=str, default="1024,512,256,64,32,8", help="Comma-separated hidden widths")
    p.add_argument("--act", type=str, default="relu", choices=["relu", "gelu", "tanh", "silu"])
    p.add_argument("--no-layernorm", action="store_true", help="Disable LayerNorm between layers")

    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cosine-tmax", type=int, default=None)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-rows", type=int, default=1, help="Skip header rows in CSV/TXT")

    a = p.parse_args()
    return TrainConfig(
        train=a.train,
        test=a.test,
        out=a.out,
        in_dim=a.in_dim,
        out_dim=a.out_dim,
        widths=a.widths,
        act=a.act,
        layernorm=(not a.no_layernorm),
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        cosine_tmax=a.cosine_tmax,
        patience=a.patience,
        min_delta=a.min_delta,
        clip_grad=a.clip_grad,
        amp=a.amp,
        num_workers=a.num_workers,
        device=a.device,
        seed=a.seed,
        skip_rows=a.skip_rows,
    )


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    cfg = parse_args()
    train(cfg)
