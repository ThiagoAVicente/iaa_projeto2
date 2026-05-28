"""Evaluate a trained model on the held-out test set.

Usage: python -m final.evaluate --model cnn

Writes metrics.json, confmat.png, roc.png, misclassified.png into
final/results/<model>/.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from final.configs import DEVICE, MODELS, RESULTS_DIR
from final.data import CLASSES, get_loaders
from final.models import BUILDERS


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        all_logits.append(model(x).cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def plot_confmat(cm, classes, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(y_true, p_pos, auc, path):
    fpr, tpr, _ = roc_curve(y_true, p_pos)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_misclassified(loader, y_true, y_pred, path, n=16):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    picks = wrong[: min(n, len(wrong))]
    # Iterate the loader once more to recover raw images for the picks.
    images, idx = [], 0
    needed = set(picks.tolist())
    for x, _ in loader:
        for i in range(x.size(0)):
            if idx in needed:
                images.append((idx, x[i]))
            idx += 1
        if len(images) == len(needed):
            break
    images.sort(key=lambda t: t[0])

    cols = 4
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (i, img) in zip(axes.flat, images):
        arr = img.numpy().transpose(1, 2, 0)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        ax.imshow(arr)
        ax.set_title(f"t={CLASSES[y_true[i]]} p={CLASSES[y_pred[i]]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(BUILDERS))
    parser.add_argument("--no-normalize", action="store_true",
                        help="evaluate the <model>_no_norm/ run")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    normalize = not args.no_normalize
    run_name = args.model + ("" if normalize else "_no_norm")
    out_dir = Path(RESULTS_DIR) / run_name
    weights = out_dir / "best.pth"
    if not weights.exists():
        raise FileNotFoundError(f"No trained weights at {weights}")

    model = BUILDERS[args.model]().to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))

    # Use stats + normalize flag from training run for exact match.
    hist_path = out_dir / "history.json"
    mean = std = None
    if hist_path.exists():
        with open(hist_path) as f:
            h = json.load(f)
        mean, std = h.get("mean"), h.get("std")
        normalize = h.get("normalize", normalize)

    _, _, test_loader = get_loaders(
        input_size=cfg["input_size"], batch_size=cfg["batch_size"],
        mean=mean, std=std, normalize=normalize,
    )

    logits, y_true = predict(model, test_loader)
    probs = torch.softmax(logits, dim=1).numpy()
    y_pred = probs.argmax(1)
    y_true = y_true.numpy()
    p_pos = probs[:, 1]  # P(class=REAL)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, p_pos)),
        "n_test": int(len(y_true)),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    plot_confmat(confusion_matrix(y_true, y_pred), CLASSES, out_dir / "confmat.png")
    plot_roc(y_true, p_pos, metrics["auc"], out_dir / "roc.png")
    plot_misclassified(test_loader, y_true, y_pred, out_dir / "misclassified.png")
    print(f"Saved plots + metrics to {out_dir}")


if __name__ == "__main__":
    main()
