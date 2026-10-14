"""
train.py

Trains SargassumConvLSTM on dl_tensor.pt / dl_tensor_metadata.pt.
Run from the same folder as model.py and dataset.py.

Design choices worth reading before changing:
  - Chronological 70/15/15 train/val/test split on sequence index (see
    dataset.py) -- val drives early stopping / LR scheduling / best-
    checkpoint selection; test is touched exactly once, at the very end,
    for the number that actually gets reported. The original snippet used
    "test" loss for checkpoint selection, which functions as a validation
    set and leaves no genuinely held-out split for final reporting.
  - Loss and every reported metric (MAE, RMSE, R2) are MASKED: land cells
    and cells whose target reading was itself missing are excluded for
    every batch, not just at evaluation time -- otherwise the model has a
    cheap way to minimize loss by matching the -1/0 sentinels over a large
    fraction of this land-heavy (10-22N) domain instead of learning actual
    Sargassum dynamics.
  - Per-channel normalization uses TRAINING-split statistics only, and
    skips the land/missing indicator channels.
  - Mixed-precision (AMP) is used on CUDA, since an RTX 3060 (Ampere) has
    full FP16 Tensor Core support -- meaningful given only 8GB VRAM. It's
    gated behind DEVICE.type == "cuda" so it's a genuine no-op on CPU.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import (
    SargassumSequenceDataset,
    build_sequence_index,
    chronological_split,
    compute_normalization_stats,
)
from model import SargassumConvLSTM

# ============================== CONFIG ======================================
TENSOR_PATH = r"C:\Users\harsh.PC\Downloads\ML project\dl_tensor.pt"
METADATA_PATH = r"C:\Users\harsh.PC\Downloads\ML project\dl_tensor_metadata.pt"

SEQ_LEN = 5                  # current + 4 lags, matching the tabular pipeline's LAG_STEPS convention
TARGET_OFFSET_DAYS = 30      # "~1 month ahead", matching AFAI_next_month elsewhere in this project
TARGET_TOLERANCE_DAYS = 10   # matching TARGET_TOLERANCE elsewhere in this project

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 0.15 is test

HIDDEN_DIM = 64
KERNEL_SIZE = (3, 3)
NUM_LAYERS = 2

BATCH_SIZE = 8        # conservative default for an 8GB RTX 3060 -- raise if VRAM headroom allows
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1000
EARLY_STOP_PATIENCE = 20
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4        # reasonable for a 6-core/12-thread Ryzen 5 7600X

SEED = 42
CHECKPOINT_PATH = r"C:\Users\harsh.PC\Downloads\ML project\sargassum_convlstm_best.pth"
LOSS_CURVE_PATH = r"C:\Users\harsh.PC\Downloads\ML project\convlstm_loss_curve.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
# =============================================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mse_loss(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff2.sum() / denom


@torch.no_grad()
def masked_metrics(pred, target, mask):
    mask_b = mask > 0.5
    if mask_b.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    p, t = pred[mask_b], target[mask_b]
    mae = (p - t).abs().mean().item()
    rmse = torch.sqrt(((p - t) ** 2).mean()).item()
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum().clamp(min=1e-8)
    r2 = (1 - ss_res / ss_tot).item()
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def run_epoch(model, loader, optimizer, scaler, train):
    model.train() if train else model.eval()
    total_loss, total_batches = 0.0, 0
    agg_pred, agg_target, agg_mask = [], [], []

    for seq, target, mask in loader:
        seq, target, mask = seq.to(DEVICE), target.to(DEVICE), mask.to(DEVICE)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred = model(seq)
                loss = masked_mse_loss(pred, target, mask)

            if train:
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        agg_pred.append(pred.detach())
        agg_target.append(target.detach())
        agg_mask.append(mask.detach())

    avg_loss = total_loss / max(1, total_batches)
    metrics = masked_metrics(torch.cat(agg_pred), torch.cat(agg_target), torch.cat(agg_mask))
    return avg_loss, metrics


def main():
    set_seed(SEED)
    log(f"Using device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    log(f"Mixed precision (AMP): {USE_AMP}")

    log("Loading tensor + metadata...")
    # weights_only=False: safe here specifically because these are files YOU generated
    # yourself (via build_tensor_pytorch.py), not checkpoints from an untrusted source --
    # PyTorch 2.6+ defaults to weights_only=True, which doesn't recognize the numpy
    # arrays (time/latitude/longitude) stored inside dl_tensor_metadata.pt.
    tensor = torch.load(TENSOR_PATH, map_location="cpu", weights_only=False)
    metadata = torch.load(METADATA_PATH, map_location="cpu", weights_only=False)
    times = pd.to_datetime(metadata["time"])
    channels = list(metadata["channels"])
    land_idx, missing_idx = channels.index("land"), channels.index("missing")
    n_time, n_lat, n_lon, n_channels = tensor.shape
    log(f"  tensor shape: {tuple(tensor.shape)} | channels: {channels}\n")

    log("Building sequence index (input window -> nearest ~1-month-ahead target)...")
    samples = build_sequence_index(n_time, times, SEQ_LEN, TARGET_OFFSET_DAYS, TARGET_TOLERANCE_DAYS)
    log(f"  {len(samples)} usable sequence samples out of {n_time} timesteps")
    if len(samples) < 50:
        log("  *** NOTE FOR THE PAPER ***: this is a small dataset for deep learning. Unlike the tabular "
            "RF/XGBoost pipeline (per-pixel rows, effectively unlimited samples), the ConvLSTM's sample "
            "count is bounded by the number of distinct TIMESTEPS in the record -- an inherent limitation "
            "worth stating explicitly, and reason for caution about overfitting despite regularization.")

    train_samples, val_samples, test_samples = chronological_split(samples, TRAIN_FRAC, VAL_FRAC)
    log(f"  train/val/test samples: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}\n")
    if not train_samples or not val_samples or not test_samples:
        raise ValueError("One of the train/val/test splits is empty -- too few total samples for this split ratio.")

    log("Computing normalization statistics from TRAINING split only...")
    norm_mean, norm_std, norm_channel_idx = compute_normalization_stats(
        tensor, times, train_samples, SEQ_LEN, channels, land_idx, missing_idx,
    )
    log(f"  normalized channels: {[channels[i] for i in norm_channel_idx]}")
    for i, c in enumerate(norm_channel_idx):
        log(f"    {channels[c]}: mean={norm_mean[i]:.4f}, std={norm_std[i]:.4f}")
    print()

    def make_ds(split_samples):
        return SargassumSequenceDataset(
            tensor, times, land_idx, missing_idx, split_samples, SEQ_LEN,
            norm_mean, norm_std, norm_channel_idx,
        )

    train_ds, val_ds, test_ds = make_ds(train_samples), make_ds(val_samples), make_ds(test_samples)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    model = SargassumConvLSTM(
        in_channels=n_channels, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, num_layers=NUM_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    log("Starting training...")
    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, scaler, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer, scaler, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1 or epoch % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            log(f"Epoch {epoch:4d}/{NUM_EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_MAE={val_metrics['MAE']:.5f} | val_R2={val_metrics['R2']:.4f} | lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                log(f"Early stopping at epoch {epoch} (no val improvement in {EARLY_STOP_PATIENCE} epochs). "
                    f"Best val_loss={best_val_loss:.6f}")
                break

    log(f"Training finished in {(time.time()-t0)/60:.1f} min\n")
    model.load_state_dict(best_state)

    log("Final evaluation (test set touched once, here only)...")
    _, train_metrics = run_epoch(model, train_loader, optimizer, scaler, train=False)
    _, val_metrics = run_epoch(model, val_loader, optimizer, scaler, train=False)
    _, test_metrics = run_epoch(model, test_loader, optimizer, scaler, train=False)

    print("\n=== Final metrics (masked: excludes land + originally-missing target cells) ===")
    for split_name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"  {split_name:5s} | MAE={m['MAE']:.5f}  RMSE={m['RMSE']:.5f}  R2={m['R2']:.5f}")

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE loss")
    plt.legend()
    plt.title("ConvLSTM training curve")
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=150)
    log(f"[saved] {LOSS_CURVE_PATH}")

    torch.save({
        "model_state_dict": best_state,
        "config": {
            "in_channels": n_channels, "hidden_dim": HIDDEN_DIM, "kernel_size": KERNEL_SIZE,
            "num_layers": NUM_LAYERS, "seq_len": SEQ_LEN,
        },
        "norm_mean": norm_mean, "norm_std": norm_std, "norm_channel_idx": norm_channel_idx,
        "channels": channels,
        "test_metrics": test_metrics,
    }, CHECKPOINT_PATH)
    log(f"[saved] {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()