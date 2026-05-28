"""Overlay ROC curves from multiple trained models in a single figure.

Loads each model's best checkpoint, runs inference on the test partition,
computes the ROC, and overlays all curves on one set of axes with AUC in
the legend.

Usage:
  python -m final.roc_combined
  python -m final.roc_combined --models cnn resnet_scratch
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from final.configs import DEVICE, MODELS, RESULTS_DIR
from final.data import get_loaders
from final.models import BUILDERS

NICE_NAMES = {
    "cnn": "CNN",
    "resnet_scratch": "ResNet-18",
    "resnet_pretrained": "ResNet-18 (pretrained)",
}


@torch.no_grad()
def predict(model, loader):
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits_all.append(model(x).cpu())
        y_all.append(y)
    return torch.cat(logits_all), torch.cat(y_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["cnn", "resnet_scratch"])
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")

    for model_name in args.models:
        cfg = MODELS[model_name]
        out_dir = Path(RESULTS_DIR) / model_name
        weights = out_dir / "best.pth"
        if not weights.exists():
            print(f"warning: {weights} missing, skipping")
            continue

        hist_path = out_dir / "history.json"
        mean = std = None
        normalize = True
        if hist_path.exists():
            with open(hist_path) as f:
                h = json.load(f)
            mean, std = h.get("mean"), h.get("std")
            normalize = h.get("normalize", True)

        model = BUILDERS[model_name]().to(DEVICE)
        model.load_state_dict(torch.load(weights, map_location=DEVICE))

        _, _, test_loader = get_loaders(
            input_size=cfg["input_size"],
            batch_size=cfg["batch_size"],
            mean=mean, std=std, normalize=normalize,
        )

        logits, y_true = predict(model, test_loader)
        probs = torch.softmax(logits, dim=1).numpy()
        p_pos = probs[:, 1]
        y_true_np = y_true.numpy()

        fpr, tpr, thresholds = roc_curve(y_true_np, p_pos)
        auc = roc_auc_score(y_true_np, p_pos)
        label = NICE_NAMES.get(model_name, model_name)
        line, = ax.plot(fpr, tpr, label=f"{label}  (AUC = {auc:.3f})")
        # Mark the operating point at threshold = 0.5.
        idx = (thresholds <= 0.5).argmax()
        ax.scatter([fpr[idx]], [tpr[idx]],
                   s=70, facecolor="white",
                   edgecolor=line.get_color(), linewidth=1.6, zorder=5)
        print(f"{label}: AUC={auc:.4f}  "
              f"at threshold 0.5: FPR={fpr[idx]:.3f}  TPR={tpr[idx]:.3f}")

    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title="ROC curves on the CIFAKE test partition",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.scatter([], [], s=70, facecolor="white", edgecolor="black",
               linewidth=1.6, label="threshold = 0.5")
    ax.legend(loc="lower right")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()

    out_path = Path(RESULTS_DIR) / "roc_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
