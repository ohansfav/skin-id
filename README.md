Skin colour starter (fresh)

This repo provides a minimal, easy-to-use pipeline to train a skin-colour predictor (continuous ITA) locally.

Quick steps:
1. Install requirements: `pip install -r requirements.txt` in your venv.
2. Prepare images and run `data_prep` to generate `patches.csv`.
3. Train with `train.py` and load the saved checkpoint with `infer.py`.

Datasets to consider:
- FairFace: https://github.com/joojs/fairface
- UTKFace: https://susanqq.github.io/UTKFace/

Pretrained backbones:
- ResNet-18 ImageNet weights are used via torchvision (downloaded automatically by the training script).

Ethics: read dataset licenses and avoid discriminatory uses.
Skin colour training starter

This repository contains a minimal starter pipeline to train a model that predicts skin colour as a continuous value (ITA - Individual Typology Angle).

Files added:
- `requirements.txt` - Python deps
- `skin_model/utils.py` - dataset helper and ITA conversion
- `skin_model/train.py` - training script using pretrained ResNet18

Recommended datasets to download before training:
- FairFace: https://github.com/joojs/fairface
- UTKFace: https://susanqq.github.io/UTKFace/
- Fitzpatrick17k (research): search "Fitzpatrick17k" — requires contacting authors

Suggested workflow:
1. Create a CSV with columns: `image_path,x,y,w,h,ita` where coordinates define a skin patch. If `ita` is empty the dataset will compute it on the patch.
2. Install dependencies from `requirements.txt`.
3. Run training:

```powershell
python -m skin_model.train --csv path\to\patches.csv --epochs 10 --batch 16
```

Model weights will be saved next to the CSV as `skin_ita_model.pt`.

Notes and ethics:
- Read dataset licenses before use. Do not use face/skin datasets for discriminatory or invasive purposes.
- Prefer continuous targets (ITA) rather than inferring protected attributes like race.

