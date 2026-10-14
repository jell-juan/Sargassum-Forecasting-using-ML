"""
dataset.py

Connects dl_tensor.pt / dl_tensor_metadata.pt (from build_tensor_pytorch.py)
to a PyTorch Dataset of (input_sequence, target, valid_mask) samples for
SargassumConvLSTM.

LEAKAGE / VALIDITY NOTES:
  - The target for a sequence ending at time t is the AFAI value at the
    NEAREST available timestep to (t + TARGET_OFFSET_DAYS), not a fixed
    number of index-steps ahead -- composite intervals aren't perfectly
    uniform, and this exactly mirrors how AFAI_next_month was defined for
    the tabular RF/XGBoost pipeline, keeping both approaches comparable.
  - valid_mask excludes (a) LAND cells at the target time and (b) cells
    whose target reading was itself originally MISSING (cloud cover etc).
    In case (b), the tensor's numeric value there is a synthetic
    placeholder (0.0) from tensor construction, not a real observation --
    training against it means training on a fabricated label, which is a
    leakage-adjacent validity problem, not just noise. Enforced HERE, at
    the point every sample is created, so it can't be forgotten downstream.
  - The chronological split (train/val/test) is done on SEQUENCE START
    INDEX, never randomly -- identical principle to the 80/20 chronological
    split used for the tabular models, for the same reason (a randomly
    assigned "test" sample could have a target that chronologically
    precedes a "train" sample's input, which is leakage).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SargassumSequenceDataset(Dataset):
    def __init__(self, tensor, times, land_idx, missing_idx, sample_indices,
                 seq_len, norm_mean=None, norm_std=None, norm_channel_idx=None):
        """
        tensor: (n_time, n_lat, n_lon, n_channels) float32, CPU.
        times: pandas.DatetimeIndex, length n_time.
        sample_indices: list of (start_idx, target_idx) tuples this split uses.
        norm_mean/std/channel_idx: from compute_normalization_stats(), or all
            None to skip normalization entirely.
        """
        self.tensor = tensor
        self.times = times
        self.land_idx = land_idx
        self.missing_idx = missing_idx
        self.sample_indices = sample_indices
        self.seq_len = seq_len
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.norm_channel_idx = norm_channel_idx

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start, target_idx = self.sample_indices[idx]
        seq = self.tensor[start:start + self.seq_len].clone()  # (seq_len, H, W, C)

        if self.norm_channel_idx is not None:
            c = self.norm_channel_idx
            seq[..., c] = (seq[..., c] - self.norm_mean) / self.norm_std

        seq = seq.permute(0, 3, 1, 2)  # (seq_len, C, H, W) -- ConvLSTM's expected per-step layout

        target = self.tensor[target_idx, :, :, 0:1].permute(2, 0, 1)  # (1, H, W), raw (unnormalized) AFAI
        land_at_target = self.tensor[target_idx, :, :, self.land_idx]
        missing_at_target = self.tensor[target_idx, :, :, self.missing_idx]
        valid_mask = ((land_at_target < 0.5) & (missing_at_target < 0.5)).unsqueeze(0).float()  # (1, H, W)

        return seq, target, valid_mask


def build_sequence_index(n_time, times, seq_len, target_offset_days, target_tolerance_days):
    """For every possible sequence start, finds the nearest available target timestep to
    (last_input_time + target_offset_days), within tolerance. Mirrors the tabular
    pipeline's AFAI_next_month matching logic exactly."""
    times = pd.DatetimeIndex(times)
    offset = pd.Timedelta(days=target_offset_days)
    tolerance = pd.Timedelta(days=target_tolerance_days)

    samples = []
    for start in range(n_time - seq_len + 1):
        last_input_idx = start + seq_len - 1
        target_time = times[last_input_idx] + offset
        deltas = np.abs((times - target_time).values)
        nearest_idx = int(deltas.argmin())
        if nearest_idx > last_input_idx and deltas[nearest_idx] <= np.timedelta64(tolerance):
            samples.append((start, nearest_idx))
    return samples


def chronological_split(samples, train_frac=0.70, val_frac=0.15):
    """Splits by SAMPLE ORDER (already chronological, since start indices increase in
    time), never randomly -- see module docstring."""
    n = len(samples)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


def compute_normalization_stats(tensor, times, train_samples, seq_len, channel_names,
                                 land_idx, missing_idx, skip_channels=("land", "missing")):
    """
    Per-channel (mean, std), computed ONLY from training-split timesteps, and ONLY over
    valid (non-land, non-missing) cells -- both to avoid leaking val/test statistics
    into training, and to avoid the land sentinel (-1.0) or the missing placeholder
    (0.0) distorting real physical channels' statistics. Binary indicator channels
    (land, missing) are skipped entirely -- normalizing a flag doesn't help and
    destroys its clean interpretability.
    """
    if not train_samples:
        raise ValueError("No training samples available to compute normalization statistics from.")

    train_time_indices = sorted({i for s, t in train_samples for i in (list(range(s, s + seq_len)) + [t])})
    channel_idx = [i for i, name in enumerate(channel_names) if name not in skip_channels]

    means, stds = [], []
    for c in channel_idx:
        vals = []
        for t_idx in train_time_indices:
            frame = tensor[t_idx, :, :, c]
            land = tensor[t_idx, :, :, land_idx]
            missing = tensor[t_idx, :, :, missing_idx]
            valid = (land < 0.5) & (missing < 0.5)
            if valid.any():
                vals.append(frame[valid])
        all_vals = torch.cat(vals)
        means.append(all_vals.mean())
        stds.append(all_vals.std().clamp(min=1e-6))

    return torch.stack(means), torch.stack(stds), channel_idx