# -*- coding: utf-8 -*-
"""
validatev2.py

Final evaluation on the held-out test set (data/test/).
- Loads best checkpoint from models/ (highest val_acc).
- Re-uses mean/std stored in the checkpoint (no recomputation).
- Reports: accuracy, classification_report (precision/recall/F1), AUC.
- Saves: confusion_matrix_test.png, gradcam_test.png, roc_curve_test.png.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TEST_PATH   = "./data/test/"
MODELS_DIR  = "models"
BATCH_SIZE  = 256
NUM_WORKERS = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Pick best checkpoint ──────────────────────────────────────────────────────
CKPT_RE = re.compile(r"best_acc([\d.]+)_ep(\d+)\.pth$")


def find_best_checkpoint():
    files = glob.glob(os.path.join(MODELS_DIR, "best_acc*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {MODELS_DIR}/")
    files.sort(
        key=lambda f: float(CKPT_RE.search(os.path.basename(f)).group(1)),
        reverse=True,
    )
    return files[0]


ckpt_path = find_best_checkpoint()
print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)

_mean = ckpt.get("mean")
_std  = ckpt.get("std")
if _mean is not None:
    _mean = _mean.cpu()
    _std  = _std.cpu()
    print(f"mean={_mean.tolist()}  std={_std.tolist()}")
else:
    print("Checkpoint trained without normalization")


# ── Model ─────────────────────────────────────────────────────────────────────
model    = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(ckpt["model_state"])
model = model.to(device).eval()


# ── Test data ─────────────────────────────────────────────────────────────────
_tx_list = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
if _mean is not None:
    _tx_list.append(transforms.Normalize(mean=_mean.tolist(), std=_std.tolist()))
transform = transforms.Compose(_tx_list)

test_set    = torchvision.datasets.ImageFolder(root=TEST_PATH, transform=transform)
class_names = test_set.classes
print(f"Classes: {class_names}  ({len(test_set)} images)")

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)


# ── Evaluate ──────────────────────────────────────────────────────────────────
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Test"):
        inputs = inputs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # P(class 1)
        preds = outputs.argmax(1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.float().cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

acc = (all_labels == all_preds).mean()
f1  = f1_score(all_labels, all_preds, average="macro")
auc = roc_auc_score(all_labels, all_probs)

print(f"\nAccuracy : {acc:.4f}")
print(f"F1 macro : {f1:.4f}")
print(f"AUC ROC  : {auc:.4f}\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


# ── Confusion matrix ──────────────────────────────────────────────────────────
cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"ResNet-18 Test Confusion Matrix (acc={acc:.4f})")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=150)
plt.close()
print("Saved confusion_matrix_test.png")


# ── ROC curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(all_labels, all_probs)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, label=f"ResNet-18 (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Test Set")
ax.legend(loc="lower right")
ax.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_test.png", dpi=150)
plt.close()
print("Saved roc_curve_test.png")


# ── GradCAM ───────────────────────────────────────────────────────────────────
def denorm(tensor_chw):
    img = tensor_chw.cpu()
    if _mean is not None:
        img = img * _std.view(3, 1, 1) + _mean.view(3, 1, 1)
    return np.clip(img.permute(1, 2, 0).numpy(), 0, 1)


real_tensor = fake_tensor = real_np = fake_np = None

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds   = outputs.argmax(1).cpu()
        for i in range(len(labels)):
            if preds[i] == labels[i]:
                lbl = labels[i].item()
                # class_names order: ['FAKE', 'REAL']  →  0=FAKE, 1=REAL
                if lbl == 1 and real_tensor is None:
                    real_tensor = images[i:i + 1].to(device)
                    real_np     = denorm(images[i])
                elif lbl == 0 and fake_tensor is None:
                    fake_tensor = images[i:i + 1].to(device)
                    fake_np     = denorm(images[i])
        if real_tensor is not None and fake_tensor is not None:
            break

target_layer = [model.layer4[-1]]

with GradCAM(model=model, target_layers=target_layer) as cam:
    real_cam = cam(input_tensor=real_tensor, targets=[ClassifierOutputTarget(1)])
    real_viz = show_cam_on_image(real_np, real_cam[0], use_rgb=True)

with GradCAM(model=model, target_layers=target_layer) as cam:
    fake_cam = cam(input_tensor=fake_tensor, targets=[ClassifierOutputTarget(0)])
    fake_viz = show_cam_on_image(fake_np, fake_cam[0], use_rgb=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(real_np);  axes[0, 0].set_title("Real - Original"); axes[0, 0].axis("off")
axes[0, 1].imshow(real_viz); axes[0, 1].set_title("Real - GradCAM");  axes[0, 1].axis("off")
axes[1, 0].imshow(fake_np);  axes[1, 0].set_title("Fake - Original"); axes[1, 0].axis("off")
axes[1, 1].imshow(fake_viz); axes[1, 1].set_title("Fake - GradCAM");  axes[1, 1].axis("off")
plt.suptitle("GradCAM — ResNet-18 (layer4[-1])", fontsize=14)
plt.tight_layout()
plt.savefig("gradcam_test.png", dpi=150)
plt.close()
print("Saved gradcam_test.png")

print(f"\nDone. Checkpoint used: {ckpt_path}")
