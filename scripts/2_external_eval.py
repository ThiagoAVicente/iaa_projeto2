"""External evaluation on Defactify_Image_Dataset.

For each generator class (SD21, SDXL, SD3, DALLE3, Midjourney) in the
Defactify test split, evaluates every trained model (cnn, resnet_scratch,
and any *_no_norm variants present under final/results/) on a balanced
subset of REAL + FAKE-from-that-generator images, then writes a single
JSON + LaTeX table summarising Acc/Prec/Rec/F1/AUC per (generator, model)
pair.

Usage:
  uv run --project final python scripts/5_external_eval.py
  uv run --project final python scripts/5_external_eval.py \
      --max-per-class 1000 --models cnn resnet_scratch
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from final.configs import DEVICE, MODELS, RESULTS_DIR  # noqa: E402
from final.data import build_transform  # noqa: E402
from final.models import BUILDERS  # noqa: E402

GENERATORS = {
    1: "SD21",
    2: "SDXL",
    3: "SD3",
    4: "DALLE3",
    5: "Midjourney",
}

HF_DATASET = "Rajarshi-Roy-research/Defactify_Image_Dataset"


def load_model(model_name, run_name):
    out_dir = Path(RESULTS_DIR) / run_name
    weights = out_dir / "best.pth"
    if not weights.exists():
        return None, None, None, None
    hist_path = out_dir / "history.json"
    mean = std = None
    normalize = True
    if hist_path.exists():
        with open(hist_path) as f:
            h = json.load(f)
        mean = h.get("mean")
        std = h.get("std")
        normalize = h.get("normalize", True)
    model = BUILDERS[model_name]().to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    cfg = MODELS[model_name]
    base_tf = build_transform(cfg["input_size"], mean, std,
                              train=False, normalize=normalize)
    # External images are not 32x32. Prepend a resize so every input
    # matches the model's expected spatial dimensions.
    transform = T.Compose([
        T.Resize((cfg["input_size"], cfg["input_size"])),
        base_tf,
    ])
    return model, transform, cfg["batch_size"], normalize


@torch.no_grad()
def predict_batch(model, batch_tensor):
    logits = model(batch_tensor.to(DEVICE, non_blocking=True))
    return F.softmax(logits, dim=1).cpu()


def collect_examples(ds, max_per_class):
    """Return {generator_id: list_of_PIL_images, 0: list_of_real_PIL_images}.

    Iterates the test split once, capping per-class collection.
    """
    bucket = {gid: [] for gid in (0, *GENERATORS)}
    target = {0: max_per_class * len(GENERATORS)}  # match the per-gen totals
    for gid in GENERATORS:
        target[gid] = max_per_class
    for row in tqdm(ds, desc="Scanning test split"):
        gid = int(row["Label_B"])
        if gid in bucket and len(bucket[gid]) < target[gid]:
            bucket[gid].append(row["Image"].convert("RGB"))
        if all(len(bucket[g]) >= target[g] for g in bucket):
            break
    return bucket


def run_model(model_name, run_name, bucket, max_per_class):
    """For one trained model, return per-generator metrics.

    Real-image pool is split evenly across the five generator subsets so
    every (generator) row sees a balanced FAKE+REAL test set.
    """
    model, transform, bs, normalize = load_model(model_name, run_name)
    if model is None:
        return None
    print(f"  -> {run_name}  (normalize={normalize})")

    real_images = bucket[0]
    # split real images evenly across generators
    per_gen = max_per_class
    real_chunks = [
        real_images[i * per_gen:(i + 1) * per_gen]
        for i in range(len(GENERATORS))
    ]

    results = {}
    for slot, (gid, gname) in enumerate(GENERATORS.items()):
        fake_imgs = bucket[gid]
        real_imgs = real_chunks[slot]
        imgs = real_imgs + fake_imgs
        # labels: REAL = 1 (positive class), FAKE = 0 (matches FAKE=0/REAL=1 ImageFolder ordering)
        labels = [1] * len(real_imgs) + [0] * len(fake_imgs)
        if not imgs:
            results[gname] = None
            continue

        # batched inference
        probs_pos = []
        for i in range(0, len(imgs), bs):
            chunk = imgs[i:i + bs]
            tensors = torch.stack([transform(im) for im in chunk])
            probs = predict_batch(model, tensors)
            probs_pos.extend(probs[:, 1].tolist())
        probs_pos = torch.tensor(probs_pos).numpy()
        y_true = torch.tensor(labels).numpy()
        y_pred = (probs_pos >= 0.5).astype(int)

        results[gname] = {
            "n_real": len(real_imgs),
            "n_fake": len(fake_imgs),
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "auc":       float(roc_auc_score(y_true, probs_pos)) if len(set(labels)) > 1 else None,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=500,
                        help="how many images per generator (and matching reals)")
    parser.add_argument("--models", nargs="+",
                        default=["cnn", "resnet_scratch"])
    parser.add_argument("--include-no-norm", action="store_true",
                        help="also evaluate *_no_norm variants if present")
    args = parser.parse_args()

    # Build run list: each tuple is (model_arch, run_dir_name)
    runs = [(m, m) for m in args.models]
    if args.include_no_norm:
        for m in args.models:
            no_norm_dir = Path(RESULTS_DIR) / f"{m}_no_norm"
            if (no_norm_dir / "best.pth").exists():
                runs.append((m, f"{m}_no_norm"))

    print(f"Loading {HF_DATASET} test split...")
    ds = load_dataset(HF_DATASET, split="test")
    print(f"Loaded {len(ds)} rows.")

    bucket = collect_examples(ds, args.max_per_class)
    for gid, imgs in bucket.items():
        name = "REAL" if gid == 0 else GENERATORS.get(gid, str(gid))
        print(f"  {name}: {len(imgs)} images")

    all_results = {}
    for arch, run_name in runs:
        print(f"Evaluating {run_name} (arch={arch})")
        all_results[run_name] = run_model(arch, run_name, bucket,
                                          args.max_per_class)

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "external_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
