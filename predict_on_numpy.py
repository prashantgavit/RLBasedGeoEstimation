from os.path import split

import numpy as np
import pickle
import torch
import pandas as pd

from torch.distributed.pipeline.sync.checkpoint import checkpoint

from classification.inference import p_key
from classification.train_base import MultiPartitioningClassifier


# Load NumPy arrays
train_array = np.load("test_data/train_array.npy")
val_array = np.load("test_data/val_array.npy")
test_array = np.load("test_data/test_array.npy")

# Load metadata (paths and IDs)
with open("test_data/paths.pkl", "rb") as f:
    paths = pickle.load(f)
    train_paths = paths["train"]
    val_paths = paths["val"]
    test_paths = paths["test"]

with open("test_data/ids.pkl", "rb") as f:
    ids = pickle.load(f)
    train_ids = ids["train"]
    val_ids = ids["val"]
    test_ids = ids["test"]

# Print shapes and details
print(f"Train array shape: {train_array.shape}")
print(f"Validation array shape: {val_array.shape}")
print(f"Test array shape: {test_array.shape}")

checkpoint = "models/base_M/epoch=014-val_loss=18.4833.ckpt"
hparams ="models/base_M/hparams.yaml"
# print("Load model from ", args.checkpoint)

paths = train_paths + val_paths + test_paths


print(paths[0])

all_country = [path.split('/')[-2].replace(" ","_") for path in paths]

train_country = [path.split('/')[-2].replace(" ","_") for path in train_paths]
val_country = [path.split('/')[-2].replace(" ","_") for path in val_paths]
test_country = [path.split('/')[-2].replace(" ","_") for path in test_paths]


all_country_set = list(set(all_country))

all_country_set.sort()



def one_hot_encode_countries(country_list, unique_countries):
    """
    One-hot encodes a list of countries based on a predefined list of unique countries.

    Parameters:
    - country_list (list): List of countries to be one-hot encoded.
    - unique_countries (list): List of all possible unique countries.

    Returns:
    - pd.DataFrame: A DataFrame containing the original country list and their one-hot encoded representation.
    """
    # Convert to DataFrame
    df = pd.DataFrame({'Country': country_list})

    # Create a DataFrame with zeros for one-hot encoding
    one_hot_encoded = pd.DataFrame(0, index=df.index, columns=unique_countries)

    # Populate one-hot encoding
    for i, country in enumerate(df['Country']):
        if country in unique_countries:  # Ensure the country exists in the predefined list
            one_hot_encoded.loc[i, country] = 1

    # Combine with the original DataFrame
    df_one_hot = pd.concat([df, one_hot_encoded], axis=1).fillna(0).set_index(['Country'])

    return df_one_hot.astype(int).values

y_train =  one_hot_encode_countries(train_country,all_country_set)
y_test  = one_hot_encode_countries(test_country,all_country_set)
y_val = one_hot_encode_countries(val_country,all_country_set)

print('y_train shape',y_train.shape)
print('y_test shape', y_test.shape)
print('y_val shape', y_val.shape)


import geopandas as gpd
from shapely.geometry import Point

def get_country_offline(lat, lon, geojson_path):
    """
    Get the country name using a GeoJSON file of country boundaries.

    Parameters:
    - lat (float): Latitude of the location.
    - lon (float): Longitude of the location.
    - geojson_path (str): Path to the GeoJSON file.

    Returns:
    - str: Country name if found, else None.
    """
    gdf = gpd.read_file(geojson_path)
    point = Point(lon, lat)  # Note: longitude first in shapely
    print(point)
    # for _, row in gdf.iterrows():
    #     if point.within(row['geometry']):
    #         return row['ADMIN']  # or row['NAME'], depending on the file
    # return None

# Example usage
# geojson_path = "../world.geo.json"
# lat, lon = 37.7749, -122.4194  # San Francisco
# country = get_country_offline(lat, lon, geojson_path)
# print(f"Country: {country}")

import reverse_geocode

def get_country(lat,lng):
    coordinates = (lat, lng),(-37.81, 144.96)
    # print(type(coordinates))
    output =  reverse_geocode.search(coordinates)

    return (output[0]['country'])


model = MultiPartitioningClassifier.load_from_checkpoint(

    checkpoint_path=str(checkpoint),
    hparams_file=str(hparams),
    map_location=None,
)

model.cuda()


train_tensor = torch.from_numpy(train_array)
val_tensor = torch.from_numpy(val_array[0:5,:]).cuda()

val_tensor = val_tensor.cuda()

pred_classes, pred_latitudes, pred_longitudes =  model.inference(val_tensor)

print('pred_latitudes',pred_classes)
rows = []
lat_lng_tuple = []
# for p_key in pred_classes.keys():
p_key ='hierarchy'
for  pred_class, pred_lat, pred_lng in zip(
        pred_classes[p_key].cpu().numpy(),
        pred_latitudes[p_key].cpu().numpy(),
        pred_longitudes[p_key].cpu().numpy(),
):
    rows.append(
        {
            # "img_id": Path(img_path).stem,
            "p_key": p_key,
            "pred_class": pred_class,
            "pred_lat": pred_lat,
            "pred_lng": pred_lng,
        }
    )
    lat_lng_tuple.append((pred_lat,pred_lng))

# print(val_tensor.shape)
# print(len(lat_lng_tuple))

# print([i['country'] for i in reverse_geocode.search(lat_lng_tuple)])

# print(a)

class task_predictor :

    def __init_(self, model):

        self.model = model


    def evaluate(self, X, y):
        X_tensor = torch.from_numpy(X).cuda()
        pred_classes, pred_latitudes, pred_longitudes = model.inference(val_tensor)

# print(f"First train path: {train_paths[0]}")
# print(f"First train ID: {train_ids[0]}")
#
# # Example usage
# print(f"First sample in train array:\n{train_array[0]}")