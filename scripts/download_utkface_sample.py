import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

OUT_DIR = Path('data/UTKFace_kaggle')
SAMPLE_DIR = Path('data/UTKFace_sample')
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

def main(n_sample=500):
    # Allow a kaggle.json placed inside the repo (./kaggle.json or ./.kaggle/kaggle.json)
    repo_dir = Path(__file__).resolve().parents[1]
    local1 = repo_dir / 'kaggle.json'
    local2 = repo_dir / '.kaggle' / 'kaggle.json'
    if local1.exists():
        os.environ['KAGGLE_CONFIG_DIR'] = str(repo_dir)
        print('Using kaggle.json from', local1)
    elif local2.exists():
        os.environ['KAGGLE_CONFIG_DIR'] = str(repo_dir / '.kaggle')
        print('Using kaggle.json from', local2)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print('Kaggle authentication failed:', e)
        print('Make sure you placed your kaggle.json in C:\\Users\\<you>\\.kaggle\\kaggle.json or in the repo as kaggle.json or ./.kaggle/kaggle.json')
        return
    dataset = 'jangedoo/utkface-new'
    print('Downloading dataset metadata...')
    listing = api.dataset_list_files(dataset)
    files = getattr(listing, 'files', None)
    # download the zip via API
    if not any(p.suffix == '.zip' for p in OUT_DIR.glob('*.zip')):
        print('Downloading dataset zip (may be large)...')
        api.dataset_download_files(dataset, path=str(OUT_DIR), unzip=False)
    else:
        print('Found existing zip in', OUT_DIR)
    # locate the zip file we just downloaded
    zips = sorted([p for p in OUT_DIR.glob('*.zip')], key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        print('No zip file found in', OUT_DIR)
        return
    zip_path = zips[0]
    print('Using zip:', zip_path)
    # extract first n_sample images
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = [n for n in z.namelist() if n.lower().endswith('.jpg')][:n_sample]
        print(f'Extracting {len(names)} images to {SAMPLE_DIR}...')
        for name in names:
            # ensure target subdirs exist
            target = SAMPLE_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(target, 'wb') as dst:
                dst.write(src.read())
    print('Done')

if __name__ == '__main__':
    main()
