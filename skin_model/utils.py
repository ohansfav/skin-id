from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path


def rgb_to_ita(rgb_array):
    arr = rgb_array.astype(np.float32) / 255.0
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    flat = arr.reshape(-1,3)
    xyz = np.dot(flat, M.T)
    X = xyz[:,0].reshape(arr.shape[0], arr.shape[1])
    Y = xyz[:,1].reshape(arr.shape[0], arr.shape[1])
    Z = xyz[:,2].reshape(arr.shape[0], arr.shape[1])
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, np.cbrt(t), (t/(3*delta**2)) + (4/29))
    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)
    L = (116 * fy) - 16
    b = 200*(fy - fz)
    ita = np.degrees(np.arctan2((L - 50), b))
    return ita


def generate_patches_from_folder(image_folder, out_csv, window=128, stride=64, skin_ratio=0.25):
    image_folder = Path(image_folder)
    files = list(image_folder.glob('**/*.jpg')) + list(image_folder.glob('**/*.png'))
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
                if patch.shape[0] < 32 or patch.shape[1] < 32:
                    continue
                r,g,b = patch[:,:,0], patch[:,:,1], patch[:,:,2]
                maxc = np.maximum(np.maximum(r,g),b)
                minc = np.minimum(np.minimum(r,g),b)
                cond = (r > 95) & (g > 40) & (b > 20) & ((maxc - minc) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
                if cond.sum() / patch.size >= skin_ratio:
                    rows.append([str(f.resolve()), int(x), int(y), int(window), int(window), ''])
    df = pd.DataFrame(rows, columns=['image_path','x','y','w','h','ita'])
    df.to_csv(out_csv, index=False)
    return out_csv
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os


def rgb_to_ita(rgb_array):
	"""Compute Individual Typology Angle (ITA) from an RGB array.
	rgb_array: HxWx3 uint8 or float in [0,255]
	Returns: HxW array of ITA degrees
	"""
	arr = rgb_array.astype(np.float32)
	# convert to linear RGB approximation (assumes sRGB gamma) — simple approximation
	arr = arr / 255.0
	# convert to XYZ using sRGB -> XYZ matrix (D65)
	M = np.array([[0.4124564, 0.3575761, 0.1804375],
				  [0.2126729, 0.7151522, 0.0721750],
				  [0.0193339, 0.1191920, 0.9503041]])
	shape = arr.shape
	flat = arr.reshape(-1,3)
	xyz = np.dot(flat, M.T)
	X = xyz[:,0].reshape(shape[0], shape[1])
	Y = xyz[:,1].reshape(shape[0], shape[1])
	Z = xyz[:,2].reshape(shape[0], shape[1])
	# avoid zeros
	Y = np.clip(Y, 1e-6, None)
	# compute L* a* b* via CIE lab approximation
	# using reference white D65
	Xn, Yn, Zn = 0.95047, 1.0, 1.08883
	def f(t):
		delta = 6/29
		return np.where(t > delta**3, np.cbrt(t), (t/(3*delta**2)) + (4/29))
	fx = f(X / Xn)
	fy = f(Y / Yn)
	fz = f(Z / Zn)
	L = (116 * fy) - 16
	# ITA = arctan((L* - 50)/b*) in degrees
	b = 200*(fy - fz)
	ita = np.degrees(np.arctan2((L - 50), b))
	return ita


class SkinPatchDataset(Dataset):
	"""A tiny dataset class that expects a CSV with columns: image_path,x,y,w,h,label
	label can be a precomputed ITA value or left empty if you want to compute on the fly.
	"""
	def __init__(self, csv_df, transform=None, label_col='ita'):
		self.df = csv_df.reset_index(drop=True)
		self.transform = transform
		self.label_col = label_col

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.loc[idx]
		img = Image.open(row['image_path']).convert('RGB')
		x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
		patch = img.crop((x, y, x+w, y+h))
		if self.transform:
			patch = self.transform(patch)
		else:
			patch = np.array(patch).astype(np.float32).transpose(2,0,1) / 255.0
			patch = torch.tensor(patch)
		if self.label_col in row and not pd.isna(row[self.label_col]):
			label = float(row[self.label_col])
		else:
			# compute ITA on central 10x10 area
			arr = np.array(patch if isinstance(patch, np.ndarray) else patch.permute(1,2,0).numpy()*255)
			h0, w0 = arr.shape[0:2]
			cx, cy = w0//2, h0//2
			s = 10
			sample = arr[cy-s:cy+s, cx-s:cx+s]
			ita_map = rgb_to_ita(sample.astype(np.uint8))
			label = float(np.nanmean(ita_map))
		return patch, torch.tensor([label], dtype=torch.float32)

