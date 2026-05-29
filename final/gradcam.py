"""Grad-CAM visualisation for the trained CNN and ResNet-18.

Picks a single set of test samples that BOTH models classify correctly
with moderate confidence, then overlays the Grad-CAM heat-map of each
model on the same input. This lets the two CAMs be compared directly
on identical images.

Restricted to the normalised model variants (cnn, resnet_scratch).

Usage:
  python -m final.gradcam
  python -m final.gradcam --n 4
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from final.configs import DEVICE, RESULTS_DIR, SEED
from final.data import CLASSES, TEST_DIR, compute_mean_std
from final.models import BUILDERS

MODEL_NAMES = ("cnn", "resnet_scratch")


def get_target_layer(model, model_name):
    """Hook the post-ReLU output of the last conv block."""
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    if model_name == "cnn":
        return model.features[-2]
    return model.layer4


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _module, _inp, out):
        self.activations = out.detach()

    def _bwd(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, target_class):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze(1).cpu().numpy()
        out = []
        for c in cam:
            m = c.max()
            out.append(c / m if m > 0 else c)
        return np.stack(out)


def load_model(model_name):
    """Returns (model, mean, std) ready for inference."""
    weights = Path(RESULTS_DIR) / model_name / "best.pth"
    hist_path = Path(RESULTS_DIR) / model_name / "history.json"
    mean, std = compute_mean_std()
    if hist_path.exists():
        with open(hist_path) as f:
            h = json.load(f)
        mean = h.get("mean", mean)
        std = h.get("std", std)
    model = BUILDERS[model_name]().to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    return model, mean, std


def shared_picks(dataset, models_info, n_per_class,
                 conf_low=0.6, conf_high=0.97):
    """Find samples where every model classifies correctly with
    confidence in [conf_low, conf_high]. Returns
    {label: [(idx, img, label, [conf_per_model])]}.
    """
    picks = {0: [], 1: []}
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            confs = []
            keep = True
            for model, mean, std in models_info:
                x = T.Normalize(mean, std)(img).unsqueeze(0).to(DEVICE)
                probs = F.softmax(model(x), dim=1)[0]
                pred = int(probs.argmax().item())
                conf = float(probs[pred].item())
                if pred != label or not (conf_low <= conf <= conf_high):
                    keep = False
                    break
                confs.append(conf)
            if keep and len(picks[label]) < n_per_class:
                picks[label].append((idx, img, label, confs))
            if all(len(picks[c]) >= n_per_class for c in picks):
                break
    return picks


def compute_cams(model, mean, std, picks, target_layer):
    cam_fn = GradCAM(model, target_layer)
    normalize = T.Normalize(mean, std)
    out = {}
    for cls in (1, 0):
        out[cls] = []
        for _, img, _, _ in picks[cls]:
            x = normalize(img).unsqueeze(0).to(DEVICE)
            out[cls].append(cam_fn(x, target_class=cls)[0])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4,
                        help="examples per class (shared across models)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = torchvision.datasets.ImageFolder(TEST_DIR, transform=T.ToTensor())

    # Load both models first so we can require agreement when picking samples.
    models = {}
    for name in MODEL_NAMES:
        model, mean, std = load_model(name)
        models[name] = {"model": model, "mean": mean, "std": std,
                        "target": get_target_layer(model, name)}
    models_info = [(d["model"], d["mean"], d["std"]) for d in models.values()]

    picks = shared_picks(dataset, models_info, args.n)
    for cls in (1, 0):
        print(f"{CLASSES[cls]}: {len(picks[cls])} shared samples picked")

    cams = {name: compute_cams(d["model"], d["mean"], d["std"],
                               picks, d["target"])
            for name, d in models.items()}

    # Layout: per class, one row of originals + one CAM row per model.
    rows_per_class = 1 + len(MODEL_NAMES)
    fig, axes = plt.subplots(rows_per_class * 2, args.n,
                             figsize=(args.n * 2 + 0.6, rows_per_class * 2 * 2.1))

    def label_axis(ax, label):
        """Show a left-side row label while hiding ticks and spines."""
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(label, fontsize=10, rotation=90, labelpad=8)

    for cls_idx, cls in enumerate((1, 0)):
        cls_name = CLASSES[cls]
        row_orig = cls_idx * rows_per_class
        for col, (_, img, _, _) in enumerate(picks[cls]):
            arr = img.numpy().transpose(1, 2, 0)
            axes[row_orig, col].imshow(arr)
            axes[row_orig, col].set_title(cls_name, fontsize=9)
            if col == 0:
                label_axis(axes[row_orig, col], "input")
            else:
                axes[row_orig, col].axis("off")
            for m_idx, name in enumerate(MODEL_NAMES):
                row = row_orig + 1 + m_idx
                cam = cams[name][cls][col]
                axes[row, col].imshow(arr)
                axes[row, col].imshow(cam, cmap="jet", alpha=0.5,
                                      vmin=0, vmax=1)
                if col == 0:
                    label_axis(axes[row, col], name)
                else:
                    axes[row, col].axis("off")

    fig.suptitle("Grad-CAM (same inputs, both models)", fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "gradcam_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
