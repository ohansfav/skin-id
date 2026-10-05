import os
import csv
from pathlib import Path
from PIL import Image
import numpy as np


def _has_cv2():
    try:
        import cv2 as _cv
        return True
    except Exception:
        return False


def _generate_with_cv2(files, face_cascade_path, min_size):
    import cv2
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    rows = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        for (x, y, w, h) in faces:
            if w < min_size or h < min_size:
                continue
            rows.append([str(f.resolve()), x, y, w, h, ''])
    return rows


def _rgb_skin_mask(arr):
    # arr shape HxWx3 uint8 RGB
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    cond = (r > 95) & (g > 40) & (b > 20) & ((maxc - minc) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
    return cond


def _generate_with_pil(files, min_size, window=128, stride=64, skin_ratio=0.25):
    rows = []
    for f in files:
        try:
            img = Image.open(f).convert('RGB')
        except Exception:
            continue
        arr = np.array(img)
        H, W = arr.shape[:2]
        for y in range(0, max(1, H - window + 1), stride):
            for x in range(0, max(1, W - window + 1), stride):
                patch = arr[y:y+window, x:x+window]
                if patch.shape[0] < min_size or patch.shape[1] < min_size:
                    continue
                mask = _rgb_skin_mask(patch)
                if mask.sum() / patch.size >= skin_ratio:
                    rows.append([str(Path(f).resolve()), int(x), int(y), int(window), int(window), ''])
    return rows


def generate_patches(image_folder, out_csv, min_size=32):
    image_folder = Path(image_folder)
    files = list(image_folder.glob('**/*.jpg')) + list(image_folder.glob('**/*.png'))
    rows = []
    if _has_cv2():
        import cv2
        HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        rows = _generate_with_cv2(files, HAAR_PATH, min_size)
    else:
        # fallback: simple RGB-based skin-window detector (no external deps)
        rows = _generate_with_pil(files, min_size)
    # write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image_path', 'x', 'y', 'w', 'h', 'ita'])
        writer.writerows(rows)
    print(f'Wrote {len(rows)} patches to {out_csv}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True, help='Folder with images')
    p.add_argument('--out', required=True, help='Output CSV path')
    args = p.parse_args()
    generate_patches(args.images, args.out)
