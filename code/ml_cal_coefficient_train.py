#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a DNN that maps a scalar input (wave steepness) to a vector of series coefficients.

Example:
    python train.py \
        --train data/train.txt \
        --eval  data/eval.txt \
        --out   outputs/run1 \
        --epochs 2000 --batch-size 64 --lr 1e-3 --hidden 64 --layers 12 --out-dim 2001

The expected data format is a plain text/CSV with rows like:
    x, a1, a2, ..., aj, K
where x is the single input feature and a1..K are the target coefficients.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_deterministic(seed: int = 42) -> None:
    """Best‑effort determinism across common libs."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_txt_as_tensors(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a whitespace or comma‑delimited text file into (X, Y) tensors.
    Expects shape [N, 1+K] with the first column the input and the rest targets.
    """
    data = np.loadtxt(path.as_posix(), delimiter=",", ndmin=2)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Data file must have at least 2 columns, got shape {data.shape} from {path}")
    x = torch.from_numpy(data[:, 0:1].astype(np.float32))
    y = torch.from_numpy(data[:, 1:].astype(np.float32))
    return x, y


# -----------------------------------------------------------------------------
# Model
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
# Training logic
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    train: Path
    eval: Optional[Path]
    out: Path
    epochs: int = 2000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 64
    layers: int = 12
    in_dim: int = 1
    out_dim: int = 2001
    patience: int = 100
    min_delta: float = 0.0
    clip_grad: float = 1.0
    cosine_tmax: Optional[int] = None  # if None, equals epochs
    amp: bool = False
    act: str = "relu"
    num_workers: int = 0
    device: Optional[str] = None  # 'cuda'/'cpu'/None(auto)
    seed: int = 42


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    x_train, y_train = load_txt_as_tensors(cfg.train)
    if y_train.shape[1] != cfg.out_dim:
        raise ValueError(
            f"Output dimension mismatch: file has {y_train.shape[1]} targets but --out-dim={cfg.out_dim}"
        )

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, len(train_ds)),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = None
    if cfg.eval is not None:
        x_val, y_val = load_txt_as_tensors(cfg.eval)
        if y_val.shape[1] != cfg.out_dim:
            raise ValueError(
                f"Output dimension mismatch (eval): file has {y_val.shape[1]} targets but --out-dim={cfg.out_dim}"
            )
        val_ds = TensorDataset(x_val, y_val)
        val_loader = DataLoader(
            val_ds,
            batch_size=min(cfg.batch_size, len(val_ds)),
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return train_loader, val_loader


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

    # Save config for reproducibility
    with (cfg.out / "config.json").open("w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}, f, indent=2)

    # Data
    train_loader, val_loader = build_dataloaders(cfg)

    # Model / Optim
    model = MLP(cfg.in_dim, cfg.out_dim, cfg.hidden, cfg.layers, cfg.act).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.cosine_tmax or cfg.epochs, eta_min=0.0
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    crit = nn.MSELoss()

    # Early stopping state
    best_val: float = math.inf
    best_epoch: int = -1
    epochs_no_improve: int = 0

    history = {"train_mse": [], "train_mae": [], "val_mse": [], "val_mae": [], "lr": []}

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Epochs"):
        model.train()
        epoch_mse = 0.0
        epoch_mae = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                pred = model(xb)
                loss = crit(pred, yb)
            scaler.scale(loss).backward()
            if cfg.clip_grad is not None and cfg.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            # Batch metrics (report as mean over batches)
            with torch.no_grad():
                mse = torch.mean((pred - yb) ** 2).item()
                mae = torch.mean(torch.abs(pred - yb)).item()
            epoch_mse += mse
            epoch_mae += mae
            n_batches += 1

        scheduler.step()
        epoch_mse /= max(1, n_batches)
        epoch_mae /= max(1, n_batches)
        history["train_mse"].append(epoch_mse)
        history["train_mae"].append(epoch_mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Validation
        val_mse, val_mae = float("nan"), float("nan")
        if val_loader is not None:
            model.eval()
            se_sum = 0.0
            ae_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = model(xb)
                    se_sum += torch.sum((pred - yb) ** 2).item()
                    ae_sum += torch.sum(torch.abs(pred - yb)).item()
                    n_val += yb.numel()
            val_mse = se_sum / n_val
            val_mae = ae_sum / n_val
            history["val_mse"].append(val_mse)
            history["val_mae"].append(val_mae)

            # Early stopping on val_mse
            improved = (best_val - val_mse) > cfg.min_delta
            if improved:
                best_val = val_mse
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "best_val_mse": best_val,
                    },
                    ckpt_dir / "best.pth",
                )
            else:
                epochs_no_improve += 1

            # Also save last
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "best_val_mse": best_val,
                },
                ckpt_dir / "last.pth",
            )

            # Early stop
            if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best @ {best_epoch}, val_mse={best_val:.6e})")
                break

        # Epoch progress print (compact)
        if val_loader is None:
            tqdm.write(f"Epoch {epoch:04d} | train_mse={epoch_mse:.3e} train_mae={epoch_mae:.3e}")
        else:
            tqdm.write(
                f"Epoch {epoch:04d} | train_mse={epoch_mse:.3e} train_mae={epoch_mae:.3e} "
                f"| val_mse={val_mse:.3e} val_mae={val_mae:.3e}"
            )

    # Save history JSON
    with (cfg.out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_mse"], label="train MSE")
        if len(history["val_mse"]) > 0:
            plt.plot(history["val_mse"], label="val MSE")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig((fig_dir / "mse_curve.png").as_posix(), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history["train_mae"], label="train MAE")
        if len(history["val_mae"]) > 0:
            plt.plot(history["val_mae"], label="val MAE")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MAE (log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig((fig_dir / "mae_curve.png").as_posix(), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(history["lr"], label="learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.tight_layout()
        plt.savefig((fig_dir / "lr.png").as_posix(), dpi=200)
        plt.close()
    except Exception as e:
        # Headless environments sometimes fail; we still want training to succeed
        print(f"Plotting failed: {e}")

    # Export predictions on eval set using best checkpoint (if available)
    if (ckpt_dir / "best.pth").exists() and (cfg.eval is not None):
        state = torch.load(ckpt_dir / "best.pth", map_location="cpu")
        model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
        model.to(device).eval()
        x_val, y_val = load_txt_as_tensors(cfg.eval)
        with torch.no_grad():
            preds = []
            for i in range(0, len(x_val), 1024):
                xb = x_val[i : i + 1024].to(device)
                yb = model(xb).cpu()
                preds.append(yb)
            pred = torch.cat(preds, dim=0)
        np.savetxt((cfg.out / "pred_eval_best.csv").as_posix(), pred.numpy(), delimiter=",")
        # Also save simple metrics
        mse = torch.mean((pred - y_val) ** 2).item()
        mae = torch.mean(torch.abs(pred - y_val)).item()
        with (cfg.out / "eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"mse": mse, "mae": mae, "best_epoch": best_epoch}, f, indent=2)
        print(f"eval MSE={mse:.6e}, MAE={mae:.6e} (best epoch {best_epoch})")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train ML series predictor (upload‑ready)")
    p.add_argument("--train", type=Path, required=True, help="Path to training txt/csv file")
    p.add_argument("--eval", type=Path, default=None, help="Optional path to eval/val txt/csv file")
    p.add_argument("--out", type=Path, required=True, help="Output directory for logs/checkpoints/figs")

    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--in-dim", type=int, default=1)
    p.add_argument("--out-dim", type=int, default=2001)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--cosine-tmax", type=int, default=None)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    p.add_argument("--act", type=str, default="relu", choices=["relu", "gelu", "tanh", "silu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Override auto device")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    return TrainConfig(
        train=args.train,
        eval=args.eval,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        layers=args.layers,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        patience=args.patience,
        min_delta=args.min_delta,
        clip_grad=args.clip_grad,
        cosine_tmax=args.cosine_tmax,
        amp=args.amp,
        act=args.act,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    # Avoid OpenMP duplicate symbol issues on some platforms
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    cfg = parse_args()
    train(cfg)
