"""Hyper-parameter sweep on a chosen scalar (learning rate or weight decay).

For each value in --values, trains a fresh model for --epochs epochs on a
(optionally smaller) subset and records the best validation accuracy.
Writes hp_sweep_<param>.json + hp_sweep_<param>.png to
final/results/<model>/.

Usage:
  python -m final.hp_sweep --model cnn
  python -m final.hp_sweep --model cnn --param weight_decay
  python -m final.hp_sweep --model cnn --values 1e-5 1e-4 1e-3 1e-2 --epochs 5
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from final.configs import DEVICE, MODELS, RESULTS_DIR, SEED
from final.data import compute_mean_std, get_loaders
from final.models import BUILDERS
from final.train import run_epoch

NICE_LABEL = {
    "lr": "learning rate",
    "weight_decay": "weight decay",
}


def sweep(model_name, param, values, epochs, subset_frac):
    cfg = MODELS[model_name]
    out_dir = Path(RESULTS_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mean, std = compute_mean_std()
    train_loader, val_loader, _ = get_loaders(
        input_size=cfg["input_size"],
        batch_size=cfg["batch_size"],
        subset_frac=subset_frac,
        mean=mean, std=std,
    )

    results = []
    for v in values:
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
        model = BUILDERS[model_name]().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        if param == "lr":
            lr, wd = v, cfg["weight_decay"]
        elif param == "weight_decay":
            lr, wd = cfg["lr"], v
        else:
            raise ValueError(f"unknown param: {param}")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

        best_val_acc = 0.0
        for ep in range(epochs):
            tr_loss, tr_acc = run_epoch(
                model, train_loader, criterion, optimizer, scaler,
                desc=f"{param}={v:.0e} ep{ep+1} train",
            )
            vl_loss, vl_acc = run_epoch(
                model, val_loader, criterion,
                desc=f"{param}={v:.0e} ep{ep+1} val",
            )
            print(f"  {param}={v:.0e}  ep{ep+1}/{epochs}  "
                  f"tr_acc={tr_acc:.4f}  vl_acc={vl_acc:.4f}")
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc

        results.append({param: v, "best_val_acc": best_val_acc})
        print(f"=> {param}={v:.0e}  best_val_acc={best_val_acc:.4f}")

    payload = {
        "model": model_name,
        "param": param,
        "epochs": epochs,
        "subset_frac": subset_frac,
        "results": results,
    }
    json_path = out_dir / f"hp_sweep_{param}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    xs = [r[param] for r in results]
    accs = [r["best_val_acc"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, accs, "o-")
    ax.set_xscale("log")
    ax.set(xlabel=NICE_LABEL.get(param, param), ylabel="best val accuracy",
           title=f"HP sweep ({param}) – {model_name}  ({epochs} epochs, "
                 f"{int(subset_frac*100)}% data)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    png_path = out_dir / f"hp_sweep_{param}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    best = max(results, key=lambda r: r["best_val_acc"])
    print(f"Best {param}={best[param]:.0e}  val_acc={best['best_val_acc']:.4f}")
    print(f"Saved {json_path}\nSaved {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(BUILDERS))
    parser.add_argument("--param", default="lr", choices=["lr", "weight_decay"])
    parser.add_argument("--values", nargs="+", type=float, default=None,
                        help="defaults to a sensible log-spaced range per param")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--subset-frac", type=float, default=0.2)
    args = parser.parse_args()

    if args.values is None:
        if args.param == "lr":
            args.values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        else:  # weight_decay
            args.values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    sweep(args.model, args.param, args.values, args.epochs, args.subset_frac)


if __name__ == "__main__":
    main()
