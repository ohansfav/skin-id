import torch
from .train import make_model
from PIL import Image
import numpy as np


def infer_image_patch(model_path, image_path, box=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    img = Image.open(image_path).convert('RGB')
    if box:
        x,y,w,h = box
        img = img.crop((x,y,x+w,y+h))
    img = img.resize((224,224))
    arr = np.array(img).astype(np.float32).transpose(2,0,1)/255.0
    import torch as _t
    t = _t.tensor(arr).unsqueeze(0).to(device)
    with _t.no_grad():
        pred = model(t).item()
    return pred
