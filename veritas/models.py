# Standard imports
import sys
import json
import torch
import numpy as np
from torch import nn
from glob import glob

# Custom packages
sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')
from vesselseg.vesselseg import networks, losses, train
#from vesselseg import networks, losses, train


class UNet(object):
    """
    UNet
    """
    def __init__(self,
                 train_params_json:str,
                 checkpoint:str,
                 device:{'cpu', 'cuda'}='cuda',
                 ):
        """
        Parameters
        ----------
        train_params_json : str
            Path to json containing the parameters that were used for training.
        checkpoint : str
            Path to checkpoint of model weights.
        device : {'cpu', 'cuda'}
            Select device to load and handle data. 
        """
        self.train_params_json=train_params_json
        self.device=device
        self.checkpoint = checkpoint
        self.model_params = json.load(open(self.train_params_json))
        # U-Net paths
        self.model = networks.SegNet(3, 1, 1, activation=None,
            backbone="UNet",
            kwargs_backbone=(self.model_params['model_architecture'])
            )
        self.losses = {0: losses.DiceLoss(labels=[1],
            activation='Sigmoid')}
        self.metrics = nn.ModuleDict({'dice': self.losses[0]})
        self.trainee = train.SupervisedTrainee(
            self.model,
            loss=self.losses,
            metrics=self.metrics)
        self.trainee = train.FineTunedTrainee.load_from_checkpoint(
            checkpoint_path=self.checkpoint,
            trainee=self.trainee,
            loss=self.losses)
        trainee = self.trainee.trainee
        trainee = trainee.to(self.device)
        trainee = trainee.eval()