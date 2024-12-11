from interface import DDPGInterface

from keras import layers
import keras
from classification.model_wrapper import taskPredictor
from classification.train_base import MultiPartitioningClassifier
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

img_shape = ( 224, 224,5, 3)

x_train = np.load("test_data/train_array.npy")
x_val = np.load("test_data/val_array.npy")
x_holdout = np.load("test_data/test_array.npy")

with open("test_data/paths.pkl", "rb") as f:
    paths = pickle.load(f)
    train_paths = paths["train"]
    val_paths = paths["val"]
    test_paths = paths["test"]

paths = train_paths + val_paths + test_paths

all_country = [path.split('/')[-2].replace(" ","").lower() for path in paths]

y_train = [path.split('/')[-2].replace(" ","").lower() for path in train_paths]
y_val = [path.split('/')[-2].replace(" ","").lower() for path in val_paths]
y_holdout = [path.split('/')[-2].replace(" ","").lower() for path in test_paths]

num_train_samples = len(y_train)
num_val_samples = len(y_val)
num_holdout_samples = len(y_holdout)


checkpoint = "models/base_M/epoch=014-val_loss=18.4833.ckpt"
hparams ="models/base_M/hparams.yaml"

model = MultiPartitioningClassifier.load_from_checkpoint(

    checkpoint_path=str(checkpoint),
    hparams_file=str(hparams),
    map_location=None,
)

task_predictor = taskPredictor(model)

from keras import layers
import keras

# Define the input shape
image_shape = (5, 3, 224, 224)


def build_actor_critic(img_shape, action_shape=(1,)):
    n_actions = action_shape[0]

    act_in = keras.Input(shape=image_shape)

    # Slice the desired crop using Lambda
    # For example, slicing the second crop (index 1) and the second channel (index 1)
    crop_act_in = layers.Lambda(lambda x: x[:, 1, 1, :, :])(act_in)  # Shape: (224, 224)

    # Reshape to add a channel dimension
    act_in_reshape = layers.Reshape((224, 224, 1))(crop_act_in)

    # Convolutional layers
    act_x = layers.Conv2D(32, (3, 3), activation='relu')(act_in_reshape)
    act_x = layers.MaxPooling2D((2, 2))(act_x)
    act_x = layers.Conv2D(64, (3, 3), activation='relu')(act_x)
    act_x = layers.MaxPooling2D((2, 2))(act_x)
    act_x = layers.Conv2D(64, (3, 3), activation='relu')(act_x)

    # Flatten and fully connected layers
    act_x = layers.Flatten()(act_x)
    act_x = layers.Dense(64, activation='relu')(act_x)
    act_x = layers.Dense(32, activation='relu')(act_x)
    act_x = layers.Dense(16, activation='relu')(act_x)

    # Output layer
    act_out = layers.Dense(n_actions, activation='sigmoid')(act_x)

    # Create the model
    actor = keras.Model(inputs=act_in, outputs=act_out)

    action_input = layers.Input(shape=(n_actions,), name='action_input')
    observation_input = layers.Input((image_shape), name='observation_input')
    observation_crop_act_in = layers.Lambda(lambda x: x[:, 1, 1, :, :])(observation_input)  # Shape: (224, 224)
    observation_act_in_reshape = layers.Reshape((224, 224, 1))(observation_crop_act_in)
    observation_x = layers.Conv2D(32, (3, 3), activation='relu')(observation_act_in_reshape)
    observation_x = layers.MaxPool2D((2, 2))(observation_x)
    observation_x = layers.Conv2D(64, (3, 3), activation='relu')(observation_x)
    observation_x = layers.MaxPool2D((2, 2))(observation_x)
    observation_x = layers.Conv2D(64, (3, 3), activation='relu')(observation_x)
    flattened_observation = layers.Flatten()(observation_x)
    x = layers.Concatenate()([action_input, flattened_observation])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1)(x)
    critic = keras.Model(inputs=[action_input, observation_input], outputs=x)

    return actor, critic, action_input


actor, critic, action_input = build_actor_critic(img_shape)
controller_batch_size = 512
task_predictor_batch_size = 256

interface = DDPGInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape,
                          custom_controller=True, actor=actor, critic=critic, action_input=action_input,
                          modify_env_params=True, modified_env_params_list=[controller_batch_size, task_predictor_batch_size])

num_train_episodes = 512

interface.train(num_train_episodes)

save_dir = 'temp'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

save_path = os.path.join(save_dir, 'pneumoniamnist_experiment_train_session')

controller_weights_save_path = save_path + 'controller_episode_' + str(num_train_episodes)
task_predictor_save_path = save_path + 'task_predictor_episode_' + str(num_train_episodes)

interface.save(controller_weights_save_path=controller_weights_save_path,
               task_predictor_save_path=task_predictor_save_path)