"""Central configuration for the final pipeline.

All knobs that affect reproducibility live here. SUBSET_FRAC defines the
single sub-sample of data used by every model so comparisons are fair.
"""

from pathlib import Path

import torch

SEED = 42
SUBSET_FRAC = 1.0
VAL_FRAC = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "final" / "results"
STATS_CACHE = RESULTS_DIR / "dataset_stats.json"

NUM_WORKERS = 8

MODELS = {
    "cnn": {
        "input_size": 32,
        "batch_size": 256,
        "lr": 1e-3,  # found with hp_sweep
        "epochs": 15,
        "weight_decay": 1e-4,
        "patience": 3,
    },
    "resnet_scratch": {
        "input_size": 32,
        "batch_size": 256,
        "lr": 3e-4,  # found with hp_sweep
        "epochs": 15,
        "weight_decay": 1e-6,  # found with hp_sweep
        "patience": 3,
    },
    "resnet_pretrained": {
        "input_size": 224,
        "batch_size": 64,
        "lr": 1e-4,
        "epochs": 5,
        "weight_decay": 1e-4,
        "patience": 2,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
