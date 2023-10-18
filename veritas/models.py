# Standard imports
import sys
import json
import torch
import numpy as np
from torch import nn
from glob import glob
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Custom packages
sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')
from vesselseg.vesselseg import networks, losses, train
from vesselseg.vesselseg.synth import SynthVesselDataset
#from vesselseg import networks, losses, train

kwargs_bb = {
    "nb_levels": 4,
    "nb_features": [
        32,
        64,
        128,
        256
    ],
    "dropout": 0,
    "nb_conv": 2,
    "kernel_size": 3,
    "activation": "ReLU",
    "norm": "instance"
}

class UnetBase(object):
    """
    Base class for UNet.
    """
    def __init__(self, version_n:int, augmentation=None, device='cuda'):
        """
        Parameters
        ----------
        version_n : int
            Version of model.
        augmentation : nn.Module
            Volume synthesis/augmentation class.
        """
        self.version_n=version_n
        self.output_path="/autofs/cluster/octdata2/users/epc28/veritas/output"
        self.version_path=f"{self.output_path}/models/version_{version_n}"
        self.json_path=f"{self.version_path}/json_params.json"
        self.checkpoint_dir = f"{self.version_path}/checkpoints"
        self.device=device
        self.backbone_dict = {}
        self.losses={0: losses.DiceLoss(labels=[1], activation='Sigmoid')}
        self.metrics=nn.ModuleDict({'dice': self.losses[0]})
        self.augmentation = augmentation


    def load_trainee(self, fine_tuned_trainee:bool=False):
        """
        Load trainee with new backbone.

        Parameters
        ----------
        fine_tuned_trainee : bool
            Make the trainee a fine tuned trainee.
        """
        self.segnet = networks.SegNet(
            ndim=3,
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            backbone='UNet',
            activation=None, 
            kwargs_backbone=self.backbone_dict
            )
        self.trainee = train.SupervisedTrainee(
            self.segnet,
            loss=self.losses[0],
            metrics=self.metrics,
            augmentation=self.augmentation
            ).to(self.device)
        if fine_tuned_trainee == True:
            pass
            #self.trainee = train.FineTunedTrainee.load_from_checkpoint(
            #    checkpoint_path="/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_8/checkpoints/epoch=135-val_loss=0.00054.ckpt",
            #    trainee=self.trainee,
            #    loss=self.losses).to(self.device)
            #self.trainee = self.trainee.eval()


    def train_it(self, data_path, train_to_val:float=0.8, batch_size:int=1):
        dataset = SynthVesselDataset(data_path)
        seed = torch.Generator().manual_seed(42)
        train_set, val_set = random_split(dataset, [train_to_val, 1-train_to_val], seed)
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size, shuffle=False)
        logger = TensorBoardLogger(self.output_path, 'models', self.version_n)
        checkpoint_callback = ModelCheckpoint(monitor="val_metric_dice", mode="min", every_n_epochs=1, save_last=True, filename='{epoch}-{val_loss:.5f}')
        trainer = Trainer(accelerator='cuda', benchmark=True, devices=1, logger=logger, callbacks=[checkpoint_callback], max_epochs=1000)
        trainer.fit(self.trainee, train_loader, val_loader)

    def prep_for_train(self):
        self.backbone_dict = {
            'nb_levels': None,
            'nb_features': None,
            'dropout': None,
            'nb_conv': None,
            'kernel_size': None,
        }
        self.load_trainee()


class NewUnet(UnetBase):
    """
    Load a Unet from a checkpoint.
    """
    def __init__(
            self,
            shape:int=256,
            nb_levels:int=4,
            nb_features:list=[32, 64, 128, 256],
            dropout:int=0,
            nb_conv:int=2,
            kernel_size:int=3,
            train_batch_size:int=2,
            val_batch_size:int=1,
            epochs:int=1000,
            switch_to_dice_epoch:int=25,
            *args,
            **kwargs
            ):
        """
        Parameters
        ----------
        shape
        nb_levels
        features
        train_batch_size
        """
        super().__init__(*args, **kwargs)
        self.shape=shape
        self.backbone_dict['nb_levels']=nb_levels
        self.backbone_dict['nb_features']=nb_features
        self.backbone_dict['dropout']=dropout
        self.backbone_dict['nb_conv']=nb_conv
        self.backbone_dict['kernel_size']=kernel_size
        self.train_batch_size=train_batch_size
        self.val_batch_size=val_batch_size
        self.epochs=epochs
        self.switch_to_dice_epoch=switch_to_dice_epoch
        self.load_trainee()

        
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