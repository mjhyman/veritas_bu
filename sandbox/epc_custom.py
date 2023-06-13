import os
import sys
import json
from glob import glob

import torch
from torch import nn
from torch.utils import data
import torch.multiprocessing as mp

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset
import numpy as np

sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas/cornucopia")
import cornucopia as cc

class CustomImageDataset(Dataset):
    def __init__(self, x_paths, y_paths, device="cpu", subset=-1, transform=None, target_transform=None):
        self.device = device
        self.x_paths = x_paths[:subset]
        self.y_paths = y_paths[:subset]

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        image = cc.LoadTransform(dtype=torch.float32, device=self.device)(self.x_paths[idx])
        label = cc.LoadTransform(dtype=torch.float32, device=self.device)(self.y_paths[idx])
        #image, label = Augmentation(64).run(image, label)
        return image, label
    
