import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(
    description="Tool for resizing the COCO dataset",
    epilog="Example usage: python resize_coco.py --size 96 --full-size-path ./data/coco/ --resized-path ./data/coco-small/"
)
parser.add_argument("--full-size-path", required=True,
                    help="Path to full size dataset")
parser.add_argument("--resized-path", required=True,
                    help="Path to save resize dataset")
parser.add_argument("--size", required=True, type=int,
                    help="Number of pixels of smaller edge after resizing")
args = parser.parse_args()

path = Path(args.full_size_path).resolve()
resized_path = Path(args.resized_path).resolve()
resized_path.mkdir(parents=True, exist_ok=True)
resize = transforms.Resize(args.size, antialias=True)

# Copy annotations
shutil.copytree(
    path / "annotations",
    resized_path / "annotations",
    dirs_exist_ok=True
)

# Resize and save all images
paths = list(path.glob("*/*.jpg"))
for p in tqdm(paths, unit="image"):
    image = Image.open(p).convert("RGB")
    image = resize(image)
    save_path = resized_path / Path(*p.parts[-2:])
    try:
        image.save(save_path)
    except FileNotFoundError:
        # Faster to ask for forgiveness
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)
