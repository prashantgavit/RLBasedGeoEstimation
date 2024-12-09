from argparse import ArgumentParser
from pathlib import Path
from math import ceil
import pandas as pd
import torch
from tqdm.auto import tqdm

# from classification.train_base import MultiPartitioningClassifier
# from classification.dataset import FiveCropImageDataset


import os
import sys
import re
from math import ceil
from typing import Dict, List, Tuple, Union
from io import BytesIO
import random
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from PIL import Image
import torchvision
import torch
import msgpack


class MsgPackIterableDatasetMultiTargetWithDynLabels(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str,
        target_mapping: Dict[str, int],
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=True,
        meta_path=None,
        cache_size=6 * 4096,
        lat_key="LAT",
        lon_key="LON",
    ):

        super(MsgPackIterableDatasetMultiTargetWithDynLabels, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

        self.meta_path = meta_path
        if meta_path is not None:
            self.meta = pd.read_csv(meta_path, index_col=0)
            self.meta = self.meta.astype({lat_key: "float32", lon_key: "float32"})
            self.lat_key = lat_key
            self.lon_key = lon_key

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary
        img = Image.open(BytesIO(x[self.key_img_encoded]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        if self.transformation is not None:
            img = self.transformation(img)

        if self.meta_path is None:
            return img, x["target"]
        else:
            _id = x[self.key_img_id].decode("utf-8")
            meta = self.meta.loc[_id]
            return img, x["target"], meta[self.lat_key], meta[self.lon_key]

    def __iter__(self):

        shard_indices = list(range(len(self.shards)))

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    # valid dataset sample?
                    _id = x[self.key_img_id].decode("utf-8")
                    try:
                        # set target value dynamically
                        if len(self.target_mapping[_id]) == 1:
                            x["target"] = self.target_mapping[_id][0]
                        else:
                            x["target"] = self.target_mapping[_id]
                    except KeyError:
                        # reject sample
                        # print(f'reject {_id} {type(_id)}')
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length


class FiveCropImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_csv: Union[str, Path, None],
        image_dir: Union[str, Path],
        img_id_col: Union[str, int] = "img_id",
        allowed_extensions: List[str] = ["jpg", "jpeg", "png"]
    ):
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        self.image_dir = image_dir
        self.img_id_col = img_id_col
        self.meta_info = None
        if meta_csv is not None:
            print(f"Read {meta_csv}")
            self.meta_info = pd.read_csv(meta_csv)
            self.meta_info.columns = map(str.lower, self.meta_info.columns)
            # rename column names if necessary to use existing data
            if "lat" in self.meta_info.columns:
                self.meta_info.rename(columns={"lat": "latitude"}, inplace=True)
            if "lon" in self.meta_info.columns:
                self.meta_info.rename(columns={"lon": "longitude"}, inplace=True)
            self.meta_info["img_path"] = self.meta_info[img_id_col].apply(
                lambda img_id: str(self.image_dir / img_id)
            )
        else:
            image_files = []
            for ext in allowed_extensions:
                image_files.extend([str(p) for p in self.image_dir.glob(f"**/*.{ext}")])
            self.meta_info = pd.DataFrame(image_files, columns=["img_path"])
            self.meta_info[self.img_id_col] = self.meta_info["img_path"].apply(
                lambda x: Path(x).stem
            )
        self.tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self):
        return len(self.meta_info.index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        meta = self.meta_info.iloc[idx]
        meta = meta.to_dict()
        meta["img_id"] = meta[self.img_id_col]

        image = Image.open(meta["img_path"]).convert("RGB")
        image = torchvision.transforms.Resize(256)(image)
        crops = torchvision.transforms.FiveCrop(224)(image)
        crops_transformed = []
        for crop in crops:
            crops_transformed.append(self.tfm(crop))
        return torch.stack(crops_transformed, dim=0), meta



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


# args = parse_args()

batch_size = 64
num_workers =  1
image_dir = 'all_data/'

dataloader = torch.utils.data.DataLoader(
    FiveCropImageDataset(meta_csv=None, image_dir= image_dir),
    batch_size=ceil(batch_size / 5),
    shuffle=True,
    num_workers=num_workers,
)

print("Loaded the data")


# print("Number of images: ", len(dataloader.dataset))
# if len(dataloader.dataset) == 0:
#     raise RuntimeError(f"No images found in {args.image_dir}")

X_list = []
image_path = []
image_id = []
count = 1

for X in tqdm(dataloader):
    # print(count)
    X_list.append(X[0])
    image_path.extend(X[1]['img_path'])
    image_id.extend(X[1]['img_id'])
    count = count+1
    # print(count)
    if count == 50:
        break


X_tensor = torch.cat(X_list, dim=0)

print(X_tensor.shape)
print("lenght of image path - ",len(image_path) )
print("length of image id  - ",len(image_id))

import numpy as np
import os
import pickle
from math import floor

# Total samples
total_samples = X_tensor.size(0)

# Calculate split sizes
train_size = floor(total_samples * 0.7)
val_size = floor(total_samples * 0.2)
test_size = total_samples - train_size - val_size

# Shuffle and split indices
indices = torch.randperm(total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Split the tensor and convert to NumPy
train_array = X_tensor[train_indices].numpy()
val_array = X_tensor[val_indices].numpy()
test_array = X_tensor[test_indices].numpy()

# Split the lists
train_paths = [image_path[i] for i in train_indices]
val_paths = [image_path[i] for i in val_indices]
test_paths = [image_path[i] for i in test_indices]

train_ids = [image_id[i] for i in train_indices]
val_ids = [image_id[i] for i in val_indices]
test_ids = [image_id[i] for i in test_indices]

# Create the parent directory
os.makedirs("test_data", exist_ok=True)

# Save splits in the parent directory as NumPy arrays
np.save("test_data/train_array.npy", train_array)
np.save("test_data/val_array.npy", val_array)
np.save("test_data/test_array.npy", test_array)

# Save metadata using pickle
with open("test_data/paths.pkl", "wb") as f:
    pickle.dump({"train": train_paths, "val": val_paths, "test": test_paths}, f)
with open("test_data/ids.pkl", "wb") as f:
    pickle.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f)

print("Data split and saved successfully in the 'test_data' directory as NumPy arrays!")