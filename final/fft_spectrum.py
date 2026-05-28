"""Mean log-magnitude 2D FFT spectrum per class.

Loads a random subsample per class, converts each image to grayscale,
computes its 2D FFT, takes the log-magnitude, then averages across the
class. Saves three heatmaps to final/results/fft_spectrum.png:
  - mean spectrum of FAKE
  - mean spectrum of REAL
  - signed difference (FAKE - REAL)

The centre of each heatmap corresponds to low spatial frequencies
(DC component, broad structure); the corners correspond to high
frequencies (fine texture, sharp edges). Generators are known to leave
characteristic high-frequency residues that are absent in real photos
(Corvi et al., 2022).
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from final.configs import RESULTS_DIR, SEED
from final.data import CLASSES, TRAIN_DIR

N_PER_CLASS = 5000


def collect_spectra(n_per_class):
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

    spectra = {}
    for c, idxs in by_class.items():
        acc = None
        for i in tqdm(idxs, desc=f"FFT {CLASSES[c]}"):
            img, _ = ds[i]
            gray = img.mean(dim=0).numpy()  # luminance proxy
            f = np.fft.fft2(gray)
            f = np.fft.fftshift(f)
            mag = np.log1p(np.abs(f))
            acc = mag if acc is None else acc + mag
        spectra[c] = acc / len(idxs)
    return spectra


def main():
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    spectra = collect_spectra(N_PER_CLASS)
    fake_label = CLASSES.index("FAKE")
    real_label = CLASSES.index("REAL")
    fake = spectra[fake_label]
    real = spectra[real_label]
    diff = fake - real

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    vmax = max(fake.max(), real.max())
    im0 = axes[0].imshow(fake, cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("FAKE  mean log-magnitude")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(real, cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("REAL  mean log-magnitude")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    dmax = np.abs(diff).max()
    im2 = axes[2].imshow(diff, cmap="seismic", vmin=-dmax, vmax=dmax)
    axes[2].set_title("FAKE - REAL  (signed)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Mean 2D FFT log-magnitude spectrum "
                 f"({N_PER_CLASS} samples per class)", fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "fft_spectrum.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Quick numeric summary: high-frequency energy (corner annulus)
    h, w = fake.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = min(h, w) // 4  # outer 3/4 of the spectrum is "high freq"
    hf_mask = dist > radius
    print(f"\nHigh-frequency mean log-magnitude (radius > {radius}px):")
    print(f"  FAKE: {fake[hf_mask].mean():.4f}")
    print(f"  REAL: {real[hf_mask].mean():.4f}")
    print(f"  diff: {(fake[hf_mask] - real[hf_mask]).mean():+.4f}")


if __name__ == "__main__":
    main()
