"""Data loaders.

One deterministic subset of `data/train/` (seeded) is shared by every model.
Mean/std are computed once on the full train split at native 32x32 and
cached to disk; all models reuse those statistics so normalization is
identical across runs.
"""

import json
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from final.configs import (
    DATA_DIR,
    NUM_WORKERS,
    SEED,
    STATS_CACHE,
    SUBSET_FRAC,
    VAL_FRAC,
)

TRAIN_DIR = Path(DATA_DIR) / "train"
TEST_DIR = Path(DATA_DIR) / "test"
CLASSES = ("FAKE", "REAL")


def compute_mean_std():
    """Compute per-channel mean/std over the full train set (native size).

    Cached to STATS_CACHE so this runs only once.
    """
    cache = Path(STATS_CACHE)
    if cache.exists():
        with open(cache) as f:
            d = json.load(f)
        return d["mean"], d["std"]

    raw = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=T.ToTensor())
    loader = DataLoader(
        raw, batch_size=512, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    for imgs, _ in tqdm(loader, desc="Computing mean/std"):
        b, _, h, w = imgs.shape
        n_pixels += b * h * w
        channel_sum += imgs.sum(dim=[0, 2, 3])
        channel_sq_sum += (imgs ** 2).sum(dim=[0, 2, 3])
    mean = (channel_sum / n_pixels).tolist()
    var = (channel_sq_sum / n_pixels) - torch.tensor(mean) ** 2
    std = var.sqrt().tolist()

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
    print(f"mean={mean}  std={std}  cached at {cache}")
    return mean, std


def build_transform(input_size, mean, std, train=False, normalize=True):
    ops = []
    if input_size != 32:
        ops.append(T.Resize((input_size, input_size)))
    if train:
        ops += [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(10)]
    ops.append(T.ToTensor())
    if normalize:
        ops.append(T.Normalize(mean, std))
    return T.Compose(ops)


def _deterministic_indices(n, frac):
    g = torch.Generator().manual_seed(SEED)
    k = int(n * frac)
    return torch.randperm(n, generator=g)[:k].tolist()


def get_loaders(input_size, batch_size, subset_frac=SUBSET_FRAC, val_frac=VAL_FRAC,
                mean=None, std=None, normalize=True):
    if normalize and (mean is None or std is None):
        mean, std = compute_mean_std()
    train_tf = build_transform(input_size, mean, std, train=True, normalize=normalize)
    eval_tf = build_transform(input_size, mean, std, train=False, normalize=normalize)

    # Two views over the same files so train indices get augmentation and
    # val indices do not.
    train_view = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    eval_view = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=eval_tf)

    idx = _deterministic_indices(len(train_view), subset_frac)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(idx), generator=g).tolist()
    cut = int(len(idx) * (1 - val_frac))
    train_idx = [idx[i] for i in perm[:cut]]
    val_idx = [idx[i] for i in perm[cut:]]

    train_set = Subset(train_view, train_idx)
    val_set = Subset(eval_view, val_idx)
    test_set = torchvision.datasets.ImageFolder(TEST_DIR, transform=eval_tf)

    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, **kw),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, **kw),
    )
