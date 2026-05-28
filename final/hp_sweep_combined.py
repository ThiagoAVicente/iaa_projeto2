"""Combine sweep results from multiple models into a single overlay plot.

Reads final/results/<model>/hp_sweep_<param>.json for each model passed via
--models and produces a single PNG hp_sweep_<param>_combined.png in
final/results/.

Usage:
  python -m final.hp_sweep_combined
  python -m final.hp_sweep_combined --param weight_decay
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

NICE_PARAM = {
    "lr": "learning rate",
    "weight_decay": "weight decay",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["cnn", "resnet_scratch"])
    parser.add_argument("--param", default="lr",
                        choices=["lr", "weight_decay"])
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4))
    for model in args.models:
        path = Path(RESULTS_DIR) / model / f"hp_sweep_{args.param}.json"
        if not path.exists():
            print(f"warning: {path} missing, skipping")
            continue
        with open(path) as f:
            payload = json.load(f)
        results = payload["results"]
        xs = [r[args.param] for r in results]
        accs = [r["best_val_acc"] for r in results]
        label = NICE_NAMES.get(model, model)
        line, = ax.plot(xs, accs, "o-", label=label)
        best = max(results, key=lambda r: r["best_val_acc"])
        ax.scatter([best[args.param]], [best["best_val_acc"]],
                   s=80, edgecolor=line.get_color(),
                   facecolor="white", zorder=5)
        print(f"{label}: best {args.param}={best[args.param]:.0e}  "
              f"val_acc={best['best_val_acc']:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel(NICE_PARAM.get(args.param, args.param))
    ax.set_ylabel("best validation accuracy")
    ax.set_title(f"{NICE_PARAM.get(args.param, args.param).title()} sweep "
                 r"(10 epochs, 20\% training subsample)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="lower center")
    fig.tight_layout()

    out_path = Path(RESULTS_DIR) / f"hp_sweep_{args.param}_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
