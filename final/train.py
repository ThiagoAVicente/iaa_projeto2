"""Train one model.

Usage: python -m final.train --model cnn
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from final.configs import DEVICE, MODELS, RESULTS_DIR, SEED
from final.data import compute_mean_std, get_loaders
from final.models import BUILDERS


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, count = 0.0, 0, 0
    for x, y in tqdm(loader, leave=False, desc=desc or ("train" if train else "eval")):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        count += x.size(0)
    return total_loss / count, correct / count


def save_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(BUILDERS))
    parser.add_argument("--subset-frac", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="resume from last checkpoint if present")
    parser.add_argument("--no-normalize", action="store_true",
                        help="skip per-channel mean/std normalization; "
                             "results saved under <model>_no_norm/")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    cfg = dict(MODELS[args.model])
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    normalize = not args.no_normalize
    run_name = args.model + ("" if normalize else "_no_norm")
    out_dir = Path(RESULTS_DIR) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.pth"

    # Stats: prefer those saved with a previous run to avoid normalization drift.
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, bad, start_epoch, prior_wall = 0.0, 0, 0, 0.0
    ckpt = None
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        mean, std = ckpt["mean"], ckpt["std"]
        history = ckpt["history"]
        best_val_acc = ckpt["best_val_acc"]
        bad = ckpt["bad"]
        start_epoch = ckpt["epoch"] + 1
        prior_wall = ckpt.get("wall_time_s", 0.0)
        print(f"Resuming from epoch {start_epoch}  best_val_acc={best_val_acc:.4f}")
    elif normalize:
        mean, std = compute_mean_std()
    else:
        mean, std = None, None

    kwargs = dict(input_size=cfg["input_size"], batch_size=cfg["batch_size"],
                  mean=mean, std=std, normalize=normalize)
    if args.subset_frac is not None:
        kwargs["subset_frac"] = args.subset_frac
    train_loader, val_loader, _ = get_loaders(**kwargs)

    model = BUILDERS[args.model]().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

    t0 = time.time()

    for epoch in range(start_epoch, cfg["epochs"]):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            desc=f"epoch {epoch+1} train",
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader, criterion, desc=f"epoch {epoch+1} val"
        )
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        print(
            f"[{epoch+1}/{cfg['epochs']}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), out_dir / "best.pth")
            bad = 0
        else:
            bad += 1

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "bad": bad,
            "history": history,
            "mean": mean,
            "std": std,
            "config": cfg,
            "normalize": normalize,
            "wall_time_s": prior_wall + (time.time() - t0),
        }, ckpt_path)

        if bad >= cfg["patience"]:
            print(f"Early stop at epoch {epoch+1}")
            break

    history["best_val_acc"] = best_val_acc
    history["wall_time_s"] = prior_wall + (time.time() - t0)
    history["config"] = cfg
    history["mean"] = mean
    history["std"] = std
    history["normalize"] = normalize
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    save_curves(history, out_dir)
    print(f"Done. best_val_acc={best_val_acc:.4f}  saved to {out_dir}")


if __name__ == "__main__":
    main()
