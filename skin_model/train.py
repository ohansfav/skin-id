import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
import numpy as np
from .utils import rgb_to_ita


class PatchDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        x,y,w,h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        patch = img.crop((x,y,x+w,y+h)).resize((224,224))
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = np.array(patch).astype(np.float32).transpose(2,0,1)/255.0
            patch = torch.tensor(patch)
        if 'ita' in row and not pd.isna(row['ita']):
            label = float(row['ita'])
        else:
            arr = np.array(patch.permute(1,2,0).numpy()*255) if isinstance(patch, torch.Tensor) else np.array(patch)*255
            label = float(np.nanmean(rgb_to_ita(arr.astype(np.uint8))))
        return patch.float(), torch.tensor([label], dtype=torch.float32)


def make_model(pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256), nn.ReLU(), nn.Linear(256,1))
    return model


def train(csv_path, epochs=5, batch_size=8, lr=1e-4, device=None, pretrained=True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])
    ds = PatchDataset(csv_path, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = make_model(pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(1)
            loss = loss_fn(pred, yb.squeeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        print(f'Epoch {epoch+1}/{epochs} - train MSE: {running/len(ds):.4f}')
    out = os.path.join(os.path.dirname(csv_path), 'skin_ita_model.pt')
    torch.save(model.state_dict(), out)
    print('Saved model to', out)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--no-pretrained', action='store_true')
    args = p.parse_args()
    train(args.csv, epochs=args.epochs, batch_size=args.batch, pretrained=not args.no_pretrained)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pandas as pd
import os
from skin_model.utils import SkinPatchDataset


def make_model(pretrained=True):
	model = models.resnet18(pretrained=pretrained)
	# replace final layer for single-value regression
	model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
							 nn.ReLU(),
							 nn.Linear(256, 1))
	return model


def train(csv_path, epochs=5, batch_size=16, lr=1e-4, device=None, pretrained=True):
	device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
	df = pd.read_csv(csv_path)
	transform = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
	])
	ds = SkinPatchDataset(df, transform=transform, label_col='ita')
	dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
	model = make_model(pretrained=pretrained).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	loss_fn = nn.MSELoss()
	for epoch in range(epochs):
		model.train()
		running = 0.0
		for xb, yb in dl:
			xb = xb.to(device).float()
			yb = yb.to(device).float()
			pred = model(xb).squeeze(1)
			loss = loss_fn(pred, yb.squeeze(1))
			opt.zero_grad()
			loss.backward()
			opt.step()
			running += loss.item() * xb.size(0)
		avg = running / len(ds)
		print(f"Epoch {epoch+1}/{epochs} - train MSE: {avg:.4f}")
	out_path = os.path.join(os.path.dirname(csv_path), 'skin_ita_model.pt')
	torch.save(model.state_dict(), out_path)
	print('Saved model to', out_path)


if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--csv', required=True, help='CSV with image_path,x,y,w,h,ita')
	p.add_argument('--no-pretrained', dest='no_pretrained', action='store_true', help='Do not use ImageNet pretrained weights')
	p.add_argument('--epochs', type=int, default=5)
	p.add_argument('--batch', type=int, default=16)
	args = p.parse_args()
	pretrained = not args.no_pretrained
	train(args.csv, epochs=args.epochs, batch_size=args.batch, device=None, pretrained=pretrained)

