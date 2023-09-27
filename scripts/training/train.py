__all__ = [
    'Train'
]
# Standard Imports
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Environment Settings
torch.no_grad()
sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')

# Custom Imports
from veritas.utils import JsonTools, PathTools
from veritas.synth import OCTSynthVesselImage
from veritas.data import OctVolume
from vesselseg.vesselseg.networks import SegNet
from vesselseg.vesselseg.losses import DiceLoss, LogitMSELoss
from vesselseg.vesselseg.train import SupervisedTrainee, FineTunedTrainee
from vesselseg.vesselseg.synth import SynthVesselDataset

################ BEGIN TRAIN PARAMETERS ################
train_params = {
    "train_new": True,
    "checkpoint": None,#"/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_7/checkpoints/last.ckpt",
    "paths": {
        "version": 9,
        "data": "/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0009",
        "vesselseg_dir": "/autofs/cluster/octdata2/users/epc28/veritas/output",
        "model": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_9",
    },
    "data": {
        "n_volumes": 10,
        "shape": 32,
        "train_to_val_ratio": 0.95
    },
    "model_architecture": {
        "nb_levels": 3,
        "nb_features": [
            8,
            8,
            16,
        ],
        "dropout": 0.1,
        "nb_conv": 2,
        "kernel_size": 3,
        "activation": "ReLU",
        "norm": "instance",
    },
    "params": {
        "train_batch_size": 1,
        "val_batch_size": 1,
        "epochs": 1000,
        "switch_to_dice_epoch": 25
    }
}
################ END TRAIN PARAMETERS ################

class Train(object):
    """
    Class for model training.
    """
    def __init__(self):
        # Number of GPU's available
        self.gpus=int(torch.cuda.device_count())
        self.trainee=None
        self.segnet=None
        self.train_loader=None
        self.val_loader=None

    def model(self, model_path:str, backbone_kwargs:dict, train_new:bool=True,
              checkpoint_path:str=None, backbone:str='UNet',
              switch_to_dice:int=25, n_dims:int=3, in_channels:int=1,
              out_channels:int=1) -> None:
        """
        Set up ML model.

        Parameters
        ----------
        model_path : str
            Path to model version directory (if training new model).
        backbone_kwargs : dict
            Kwargs for defining backbone parameters (architecture) of model.
        train_new : bool
            Whether to train a new U-Net from scratch.
        checkpoint_path : str
            Path to checkpoint if retraining.
        backbone : {'UNet', 'MeshNet', 'ATrousNet'}
            Backbone of ML model.
        switch_to_dice : int
            Epoch at which loss function is switched to dice.
        n_dims : int
            Number of dimensions in input data.
        in_channels : int
            Number of channels in input.
        out_channels : int
            Number of channels in output.

        Returns
        -------
        self.FTtrainee
            Trainee whose loss function changes after N epochs.
        self.segnet
            Instantiated segmentation neural network.
        """
        # Instantiate segnet
        self.segnet = SegNet(
            ndim=n_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None,
            backbone=backbone,
            kwargs_backbone=(backbone_kwargs)
            )
        # Loss functions
        self.losses = {
            0: LogitMSELoss(labels=[1]),
            switch_to_dice: DiceLoss(labels=[1], activation='Sigmoid')
        }
        # Metrics
        self.metrics = torch.nn.ModuleDict({
            'logitmse': self.losses[0],
            'dice': self.losses[switch_to_dice]
            })
        # Instantiate vessel synth
        self.synth = OCTSynthVesselImage()
        # Instantiate supervised trainee
        self.trainee = SupervisedTrainee(
            network=self.segnet,
            loss=self.losses[0],
            augmentation=self.synth,
            metrics=self.metrics
            )
        
        if train_new == True:
            PathTools(path=model_path).makeDir()
            JsonTools().log(
                train_params, f"{train_params['paths']['model']}/train_params.json"
                )
            self.FTtrainee = FineTunedTrainee(
                trainee=self.trainee,
                loss=self.losses
                )
        elif train_new == False:
            checkpoint = checkpoint_path
            self.FTtrainee = FineTunedTrainee.load_from_checkpoint(
                checkpoint_path=checkpoint,
                trainee=self.trainee,
                loss=self.losses
                )
    

    def dataLoaders(self,
                    data_dir:str,
                    n_volumes:int,
                    train_to_val_ratio:float=0.8,
                    train_batch_size:int=2,
                    val_batch_size:int=1,
                    device:str="cuda",
                    synthetic:bool=True
                    ) -> None:
        """
        Instantiate train and validation data loaders.

        Parameters
        ----------
        data_dir : str
            Location of directory containing synthetic vascular networks.
        n_volumes : int
            Number of synthetic vascular networks to load.
        train_to_val_ratio : float
            Ratio of training data to validation data that should be used.
        train_batch_size : int
            Number of vascular networks that are used for training at once.
        val_batch_size : int
            Number of vascular networks that are used for validating at once.
        device : {'cpu', 'cuda'}
            Select device to load and handle data. 
        synthetic : bool, default True
            sets up the train and val loaders to use real or synthetic data.

        Returns
        -------
        self.train_loader
            Loader for training data.
        self.val_loader
            Loader for validation data.
        """
        seed = torch.Generator().manual_seed(42)
        if synthetic ==  True:
            dataset = SynthVesselDataset(
                data_dir,
                subset=slice(n_volumes), device=device
                )
        elif synthetic == False:
            dataset = OctVolume(device=device, subset=slice(n_volumes))
        else:
            print("I don't know what to do with the data!")
        # Calculate number of elements in training data set 
        n_train = int(n_volumes // (1 / train_to_val_ratio))
        # Calculate number of elements in validation data set
        n_val = int(n_volumes - n_train)
        # Make training and validation sets
        train_set, val_set = torch.utils.data.random_split(dataset,
            [n_train, n_val], generator=seed)
        # Instantiate train loader
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=train_batch_size, shuffle=True
            )
        # Instantiate validation loader
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=val_batch_size, shuffle=False
            )
        self.train_loader = train_loader
        self.val_loader = val_loader

    
    def train(self,
              save_dir:str,
              default_root_dir:str,
              version:int,
              name:str='models',
              device:str='cuda',
              max_epochs:int=1000,
            ) -> Trainer:
        """
        Set up trainer.

        Parameters
        ----------
        save_dir : str
            Directory to save models.
        default_root_dir : str
            default root directory.
        version : int
            Experiment/model version number.
        name : str
            Experiment name.
        device : {'cpu', 'cuda'}
            Device used for training.
        max_epochs : int
            Maximum number of epochs to train model.

        """
        # Remove events log and last checkpoint
        PathTools(train_params["paths"]["model"]).patternRemove("*events.out*")
        PathTools(train_params["paths"]["model"]).patternRemove("*last.ckpt*")
        # Instantiate logger
        logger = TensorBoardLogger(
            save_dir=save_dir, name=name, version=version
            )
        checkpoint_callback = ModelCheckpoint(monitor="val_metric_dice",
            mode="min", every_n_epochs=1, save_last=True,
            filename='{epoch}-{val_loss:.5f}'
            )
        
        if self.gpus == 1:
            trainer = Trainer(accelerator=device, benchmark=True, devices=1,
            logger=logger, callbacks=[checkpoint_callback],
            max_epochs=max_epochs
            )
            trainer.fit(self.FTtrainee, self.train_loader, self.val_loader)

        elif self.gpus > 1:
            import torch.multiprocessing as mp
            print(f"Training on {self.gpus} gpus")
            trainer_ = Trainer(
                accelerator=device,
                enable_progress_bar=True,
                detect_anomaly=True,
                default_root_dir=default_root_dir,
                accumulate_grad_batches=5,
                devices=self.gpus,
                strategy="ddp",
                num_nodes=1,
                callbacks=[checkpoint_callback],
                logger=logger,
                log_every_n_steps=1,
                max_epochs=max_epochs
                )
            mp.set_start_method('spawn', force=True)
            n_processes = self.gpus
            self.segnet.share_memory()
            processes = []
            for rank in range(n_processes):
                process = mp.Process(target=trainer_.fit(
                    self.FTtrainee,
                    self.train_loader,
                    self.val_loader),
                    args=(self.segnet,))
                process.start()
                processes.append(process)
            for proc in processes:
                proc.join()

    def main(self):
        """
        Train model.
        """
        self.model(
            model_path=train_params['paths']['model'],
            backbone_kwargs=train_params['model_architecture'],
            train_new=train_params["train_new"],
            checkpoint_path=train_params["checkpoint"],
            switch_to_dice=train_params['params']['switch_to_dice_epoch']
            )
        self.dataLoaders(
            data_dir=train_params['paths']['data'],
            n_volumes=train_params['data']['n_volumes'],
            train_to_val_ratio=train_params['data']['train_to_val_ratio'],
            train_batch_size=train_params['params']['train_batch_size'],
            val_batch_size=train_params['params']["val_batch_size"],
        )
        self.train(
            save_dir=train_params['paths']['vesselseg_dir'],
            default_root_dir=train_params['paths']['vesselseg_dir'],
            version=train_params['paths']['version'],
            max_epochs=train_params['params']['epochs']
        )


if __name__ == "__main__":
    unet = Train()
    unet.main()