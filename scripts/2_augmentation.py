"""
2_augmentation.py

Applies augmentations to data/train/FAKE/ and data/train/REAL/.
Adds hflip, rot+10, rot-10 alongside originals. Originals untouched.
Run from project root: python scripts/2_augmentation.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

SRC_ROOT = Path("data/train")
CLASSES  = ["FAKE", "REAL"]
EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANGLE    = 10


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img_rgb, path):
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def hflip(img):
    return cv2.flip(img, 1)


def rotate(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)


def main():
    for cls in CLASSES:
        src_dir = SRC_ROOT / cls
        files = [f for f in src_dir.iterdir() if f.suffix.lower() in EXTS]
        print(f"{cls}: {len(files)} originals → {len(files) * 3} new images")

        for f in tqdm(files, desc=cls):
            img = load_image(f)
            save_image(hflip(img),           src_dir / f"{f.stem}_hflip{f.suffix}")
            save_image(rotate(img,  ANGLE),  src_dir / f"{f.stem}_rot+{ANGLE}{f.suffix}")
            save_image(rotate(img, -ANGLE),  src_dir / f"{f.stem}_rot-{ANGLE}{f.suffix}")

    print("Done.")


if __name__ == "__main__":
    main()
