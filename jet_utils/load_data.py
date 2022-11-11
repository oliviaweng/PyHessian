import os
import time
import shutil
import logging
from datetime import datetime
import json
import yaml

import numpy as np

import torch
import torch.optim
import torch.utils.data

from jet_utils.jet_dataset import JetTaggingDataset


# ------------------------------------------------------------
# dataset
# ------------------------------------------------------------
def open_config(filename):
    with open(filename, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_data(path, batch_size, data_percentage, config, shuffle=True):
    dataset = JetTaggingDataset(path, config["features"], config["preprocess"])
    print(len(dataset))
    dataset_length = int(len(dataset) * data_percentage)
    partial_dataset, _ = torch.utils.data.random_split(
        dataset, [dataset_length, len(dataset) - dataset_length]
    )

    data_loader = torch.utils.data.DataLoader(
        partial_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    return data_loader


def load_dataset(data_path, batch_size, config):
    data_config = open_config(config)["data"]
    train_loader = load_data(
        path=os.path.join(data_path, "train"),
        batch_size=batch_size,
        data_percentage=0.75,
        config=data_config,
    )
    val_loader = load_data(
        path=os.path.join(data_path, "val"),
        batch_size=batch_size,
        data_percentage=1,
        shuffle=False,
        config=data_config,
    )
    return train_loader, val_loader

