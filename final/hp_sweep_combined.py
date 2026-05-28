"""Combine learning-rate sweep results from multiple models into a single
overlay plot.

Reads final/results/<model>/hp_sweep_lr.json for each model passed via
--models (default: cnn resnet_scratch) and produces a single PNG
hp_sweep_combined.png in final/results/.

Usage:
  python -m final.hp_sweep_combined
  python -m final.hp_sweep_combined --models cnn resnet_scratch
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from final.configs import RESULTS_DIR

NICE_NAMES = {
    "cnn": "CNN",
    "resnet_scratch": "ResNet-18",
    "resnet_pretrained": "ResNet-18 (pretrained)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["cnn", "resnet_scratch"])
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4))
    for model in args.models:
        path = Path(RESULTS_DIR) / model / "hp_sweep_lr.json"
        if not path.exists():
            print(f"warning: {path} missing, skipping")
            continue
        with open(path) as f:
            payload = json.load(f)
        results = payload["results"]
        lrs = [r["lr"] for r in results]
        accs = [r["best_val_acc"] for r in results]
        label = NICE_NAMES.get(model, model)
        line, = ax.plot(lrs, accs, "o-", label=label)
        best = max(results, key=lambda r: r["best_val_acc"])
        ax.scatter([best["lr"]], [best["best_val_acc"]],
                   s=80, edgecolor=line.get_color(),
                   facecolor="white", zorder=5)
        print(f"{label}: best lr={best['lr']:.0e}  "
              f"val_acc={best['best_val_acc']:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("best validation accuracy")
    ax.set_title("Learning-rate sweep (10 epochs, 20\\% training subsample)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="lower center")
    fig.tight_layout()

    out_path = Path(RESULTS_DIR) / "hp_sweep_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
