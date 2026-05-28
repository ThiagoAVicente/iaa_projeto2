"""Sample grid visualization.

Outputs two figures:
  * sample_grid.png      - raw images, [0, 1] range (what the dataset looks like)
  * sample_grid_norm.png - same images after per-channel mean/std normalization
                          (what the model sees), displayed with min-max scaling
                          so the structure stays visible.

Usage: python -m final.sample_grid
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T

from final.configs import RESULTS_DIR, SEED
from final.data import TRAIN_DIR, CLASSES, compute_mean_std

N_PER_CLASS = 6


def collect(n_per_class):
    rng = random.Random(SEED)
    raw = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=T.ToTensor())
    by_class = {0: [], 1: []}
    indices = list(range(len(raw)))
    rng.shuffle(indices)
    for i in indices:
        _, label = raw.samples[i]
        if len(by_class[label]) < n_per_class:
            by_class[label].append(i)
        if all(len(v) == n_per_class for v in by_class.values()):
            break
    images = {cls: torch.stack([raw[i][0] for i in by_class[cls]])
              for cls in by_class}
    return images


def plot_grid(images, title, path):
    """images: {label: tensor[N, 3, H, W]}.

    Top row REAL, bottom row FAKE (matches alphabetical FAKE=0, REAL=1).
    """
    n = next(iter(images.values())).shape[0]
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.6))
    label_idx = {"REAL": CLASSES.index("REAL"), "FAKE": CLASSES.index("FAKE")}
    rows = [("REAL", label_idx["REAL"]), ("FAKE", label_idx["FAKE"])]
    for r, (name, cls) in enumerate(rows):
        for c in range(n):
            img = images[cls][c].numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, loc="left", fontsize=11)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_imgs = collect(N_PER_CLASS)
    plot_grid(raw_imgs, "Raw samples (range [0, 1])",
              out_dir / "sample_grid.png")

    mean, std = compute_mean_std()
    norm = T.Normalize(mean, std)
    norm_imgs = {cls: torch.stack([norm(img) for img in batch])
                 for cls, batch in raw_imgs.items()}
    plot_grid(norm_imgs,
              f"After Normalize(mean={[round(m,3) for m in mean]}, "
              f"std={[round(s,3) for s in std]})",
              out_dir / "sample_grid_norm.png")


if __name__ == "__main__":
    main()
