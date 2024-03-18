import glob
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as audio_transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None, mode="train"):

        super().__init__()
        self.transform = transform
        self.mode = mode
        self.segments = glob.glob(os.path.join(data_root, "*.npz"))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        segment_path = self.segments[index]
        data = np.load(segment_path, allow_pickle=True)
        mae = data["mae_features"]
        labels = data["labels"]
        return mae, labels


def get_loader(cfg):

    train_dataset = CustomDataset(cfg.Data.train_data_root)
    valid_dataset = CustomDataset(cfg.Data.val_data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.Data.loader.batch_size,
        num_workers=cfg.Data.loader.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    if cfg.Data.loader.test_batch_size:
        test_batch_size = cfg.Data.loader.test_batch_size
    else:
        test_batch_size = cfg.Data.loader.batch_size
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=test_batch_size,
        num_workers=cfg.Data.loader.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader
