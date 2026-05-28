"""Per-channel pixel-intensity histograms by class.

Loads a fixed-size random subsample from the training partition, separates
samples by class, computes per-channel histograms of pixel values in [0,1],
and saves a 1x3 figure (one panel per channel) overlaying FAKE vs REAL.

Usage: python -m final.pixel_hist
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from final.configs import RESULTS_DIR, SEED
from final.data import CLASSES, TRAIN_DIR

N_PER_CLASS = 5000
BINS = 50
CHANNEL_NAMES = ("Red", "Green", "Blue")


def collect_by_class(n_per_class):
    rng = random.Random(SEED)
    ds = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=T.ToTensor())
    by_class = {c: [] for c in range(len(CLASSES))}
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for i in indices:
        _, label = ds.samples[i]
        if len(by_class[label]) < n_per_class:
            by_class[label].append(i)
        if all(len(v) == n_per_class for v in by_class.values()):
            break
    pixels = {c: [] for c in by_class}
    for c, idxs in by_class.items():
        for i in tqdm(idxs, desc=CLASSES[c]):
            img, _ = ds[i]
            pixels[c].append(img.numpy())
    return {c: np.stack(arr) for c, arr in pixels.items()}


def main():
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect_by_class(N_PER_CLASS)
    fake_label = CLASSES.index("FAKE")
    real_label = CLASSES.index("REAL")
    fake = data[fake_label]  # (N, 3, H, W)
    real = data[real_label]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    bin_edges = np.linspace(0, 1, BINS + 1)
    for ch, name in enumerate(CHANNEL_NAMES):
        ax = axes[ch]
        ax.hist(fake[:, ch].ravel(), bins=bin_edges, alpha=0.55,
                label="FAKE", density=True, color="#d62728")
        ax.hist(real[:, ch].ravel(), bins=bin_edges, alpha=0.55,
                label="REAL", density=True, color="#1f77b4")
        ax.set(xlabel="pixel intensity", title=f"{name} channel")
        if ch == 0:
            ax.set_ylabel("density")
        ax.grid(True, ls="--", alpha=0.4)
    axes[-1].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(f"Per-channel pixel intensity distribution  "
                 f"({N_PER_CLASS} samples/class)", fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "pixel_hist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Per-channel summary stats (printable in report)
    print("\nChannel  | class | mean   | std    | median")
    print("-" * 48)
    for ch, name in enumerate(CHANNEL_NAMES):
        for label, arr in [("FAKE", fake), ("REAL", real)]:
            v = arr[:, ch]
            print(f"{name:8s} | {label:5s} | {v.mean():.4f} | {v.std():.4f} | {np.median(v):.4f}")


if __name__ == "__main__":
    main()
