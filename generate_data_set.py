from argparse import ArgumentParser
from pathlib import Path
from math import ceil
import pandas as pd
import torch
from tqdm.auto import tqdm

from classification.train_base import MultiPartitioningClassifier
from classification.dataset import FiveCropImageDataset


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/base_M/epoch=014-val_loss=18.4833.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams",
        type=Path,
        default=Path("models/base_M/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        default=Path("all_data"),
        help="Folder containing images. Supported file extensions: (*.jpg, *.jpeg, *.png)",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()


args = parse_args()

dataloader = torch.utils.data.DataLoader(
    FiveCropImageDataset(meta_csv=None, image_dir=args.image_dir),
    batch_size=ceil(args.batch_size / 5),
    shuffle=False,
    num_workers=args.num_workers,
)

print("Number of images: ", len(dataloader.dataset))
if len(dataloader.dataset) == 0:
    raise RuntimeError(f"No images found in {args.image_dir}")

X_list = []
metadata = []
for X in tqdm(dataloader):
    X_list.append(X[0])
    metadata.append(X[1])
