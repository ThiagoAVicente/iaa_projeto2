# -*- coding: utf-8 -*-
"""
train_iter.py

Iterative trainer with checkpointing.

Behaviour:
  - Scans `models/` for existing checkpoints.
  - No checkpoint -> fresh ResNet-18, fresh history, fresh optimizer state.
  - Checkpoint found -> loads best (highest val_acc), resumes training.
  - After each epoch, validates. If new val_acc beats previous best,
    deletes the old best file and saves a new one.
  - History (loss + accuracy per epoch) is persisted inside checkpoint
    so curves accumulate across runs.
  - On exit: saves training_curves.png, confusion_matrix.png,
    and prints classification report.

Run repeatedly:  python train_iter.py
Each invocation trains EPOCHS_THIS_RUN more epochs.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
PATH        = "./data/"
TRAIN_PATH  = PATH + "train/"
MODELS_DIR  = "models"

PRETRAINED       = False
NORMALIZE        = True  # False = no Z-score; useful sanity check on low-res inputs
NUM_WORKERS      = 16    # set to 2 on Colab free tier
SEED             = 42
BATCH_SIZE       = 256
LR               = 1e-4
EPOCHS_THIS_RUN  = 3    # how many extra epochs to add this invocation
PATIENCE         = 3    # stop if val_acc doesn't improve for this many epochs in a row

os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


# ── Checkpoint discovery ──────────────────────────────────────────────────────
CKPT_RE = re.compile(r"best_acc([\d.]+)_ep(\d+)\.pth$")


def find_best_checkpoint():
    files = glob.glob(os.path.join(MODELS_DIR, "best_acc*.pth"))
    if not files:
        return None
    files.sort(
        key=lambda f: float(CKPT_RE.search(os.path.basename(f)).group(1)),
        reverse=True,
    )
    return files[0]


# ── Mean/std (computed once, then cached in checkpoint) ───────────────────────
def compute_mean_std(path):
    _raw = torchvision.datasets.ImageFolder(root=path, transform=transforms.ToTensor())
    _loader = torch.utils.data.DataLoader(
        _raw, batch_size=512, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    _mean = torch.zeros(3, device=device)
    _std  = torch.zeros(3, device=device)
    for imgs, _ in tqdm(_loader, desc="Computing mean/std"):
        imgs = imgs.to(device, non_blocking=True)
        _mean += imgs.mean(dim=[0, 2, 3])
        _std  += imgs.std(dim=[0, 2, 3])
    _mean /= len(_loader)
    _std  /= len(_loader)
    return _mean.cpu(), _std.cpu()


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model():
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(device)


# ── Load or init ──────────────────────────────────────────────────────────────
model     = build_model()
optimizer = optim.Adam(model.parameters(), lr=LR)
scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
criterion = nn.CrossEntropyLoss()

best_ckpt_path = find_best_checkpoint()

if best_ckpt_path is None:
    print("No checkpoint found, fresh start.")
    if NORMALIZE:
        _mean, _std = compute_mean_std(TRAIN_PATH)
    else:
        _mean, _std = None, None
        print("NORMALIZE=False -> skipping mean/std computation")
    history        = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "val_f1":     [], "val_auc": [],
    }
    start_epoch              = 0
    best_val_acc             = 0.0
    epochs_since_improvement = 0
else:
    print(f"Resuming from {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    history      = ckpt["history"]
    history.setdefault("val_f1", [])   # back-compat for old checkpoints
    history.setdefault("val_auc", [])
    start_epoch              = ckpt["epoch"]
    best_val_acc             = ckpt["best_val_acc"]
    epochs_since_improvement = ckpt.get("epochs_since_improvement", 0)
    _mean        = ckpt["mean"]
    _std         = ckpt["std"]

if _mean is not None:
    print(f"Normalization mean={_mean.tolist()}  std={_std.tolist()}")
else:
    print("Normalization disabled")
print(f"Resuming at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")


# ── Data ──────────────────────────────────────────────────────────────────────
_tx_list = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
if _mean is not None:
    _tx_list.append(transforms.Normalize(mean=_mean.tolist(), std=_std.tolist()))
transform = transforms.Compose(_tx_list)

dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
class_names = dataset.classes

train_data, val_data = torch.utils.data.random_split(
    dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(SEED)
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)


# ── Epoch loops ───────────────────────────────────────────────────────────────
def run_train_epoch(epoch_label):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"{epoch_label} train")
    for inputs, labels in loop:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * inputs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += inputs.size(0)
        loop.set_postfix(loss=total_loss / total, acc=correct / total)
    return total_loss / total, correct / total


def run_val_epoch(epoch_label):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []
    loop = tqdm(val_loader, desc=f"{epoch_label} val")
    with torch.no_grad():
        for inputs, labels in loop:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            probs       = torch.softmax(outputs, dim=1)[:, 1]   # P(class 1 = REAL)
            preds       = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())
            loop.set_postfix(loss=total_loss / total, acc=correct / total)
    return (
        total_loss / total, correct / total,
        np.array(all_labels), np.array(all_preds), np.array(all_probs),
    )


# ── Train ─────────────────────────────────────────────────────────────────────
last_val_labels, last_val_preds, last_val_probs = None, None, None

for i in range(EPOCHS_THIS_RUN):
    epoch = start_epoch + i + 1
    label = f"Epoch {epoch}"

    train_loss, train_acc = run_train_epoch(label)
    val_loss, val_acc, last_val_labels, last_val_preds, last_val_probs = run_val_epoch(label)
    val_f1  = f1_score(last_val_labels, last_val_preds, average="macro")
    val_auc = roc_auc_score(last_val_labels, last_val_probs)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["val_auc"].append(val_auc)

    print(
        f"{label}  train_loss={train_loss:.4f} acc={train_acc:.4f}  "
        f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f}"
    )

    if val_acc > best_val_acc:
        epochs_since_improvement = 0
        prev_best = find_best_checkpoint()
        if prev_best is not None and os.path.exists(prev_best):
            os.remove(prev_best)
            print(f"Removed previous best: {prev_best}")
        best_val_acc = val_acc
        new_path = os.path.join(MODELS_DIR, f"best_acc{val_acc:.4f}_ep{epoch}.pth")
        torch.save(
            {
                "model_state":              model.state_dict(),
                "optimizer_state":          optimizer.state_dict(),
                "scaler_state":             scaler.state_dict(),
                "history":                  history,
                "epoch":                    epoch,
                "best_val_acc":             best_val_acc,
                "epochs_since_improvement": epochs_since_improvement,
                "mean":                     _mean,
                "std":                      _std,
            },
            new_path,
        )
        print(f"Saved new best: {new_path}")
    else:
        epochs_since_improvement += 1
        print(f"No improvement ({epochs_since_improvement}/{PATIENCE}). best={best_val_acc:.4f}")
        if epochs_since_improvement >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break


# ── Plots + report ────────────────────────────────────────────────────────────
epochs_axis = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(epochs_axis, history["train_loss"], label="Train", marker="o")
axes[0].plot(epochs_axis, history["val_loss"],   label="Validation", marker="o")
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_axis, history["train_acc"], label="Train Acc", marker="o")
axes[1].plot(epochs_axis, history["val_acc"],   label="Val Acc",   marker="o")
if history.get("val_f1"):
    axes[1].plot(epochs_axis[-len(history["val_f1"]):], history["val_f1"],
                 label="Val F1 (macro)", marker="s", linestyle="--")
if history.get("val_auc"):
    axes[1].plot(epochs_axis[-len(history["val_auc"]):], history["val_auc"],
                 label="Val AUC", marker="^", linestyle=":")
axes[1].set_title("Accuracy / F1 per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()
print("Saved training_curves.png")

if last_val_labels is not None:
    print("\nClassification Report (last validation epoch):")
    print(classification_report(last_val_labels, last_val_preds, target_names=class_names))

    cm   = confusion_matrix(last_val_labels, last_val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("ResNet-18 Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")

print(f"\nBest val_acc so far: {best_val_acc:.4f}")
