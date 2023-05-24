import os
import sys
import json

import torch
from torch import nn
from torch.utils import data
import torch.multiprocessing as mp

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append("vesselseg")
from vesselseg.networks import SegNet
from vesselseg.losses import DiceLoss, LogitMSELoss
from vesselseg.train import SupervisedTrainee, FineTunedTrainee
from vesselseg.synth import SynthVesselDataset, SynthVesselImage

os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
retrain_params = json.load(open("vesselseg/scripts/retraining/retrain_params.json"))

class Train(object):

    def __init__(self):
        self.version_path = f"{retrain_params['paths']['vesselseg_dir']}/models/version_{retrain_params['paths']['version']}"
        self.last_checkpoint = f"{self.version_path}/checkpoints/last.ckpt"
        self.train_params = json.load(open(f"{self.version_path}/train_params.json"))

        self.logger = TensorBoardLogger(self.train_params['paths']['vesselseg_dir'], name="models", version=self.train_params['paths']['version'])
        self.gpus = torch.cuda.device_count()
        self.seed = torch.Generator().manual_seed(42)
        self.n_volumes = self.train_params['data']['n_volumes']
        self.dataset = SynthVesselDataset(self.train_params['paths']['data'], subset=slice(self.n_volumes), device="cuda")
        self.synth = SynthVesselImage()
        self.segnet = SegNet(3, 1, 1, activation=None, backbone='UNet',kwargs_backbone=(self.train_params['model_architecture']))
        self.losses = {0: LogitMSELoss(labels=[1]), self.train_params['params']['switch_to_dice_epoch']: DiceLoss(labels=[1], activation='Sigmoid')}
        self.metrics = nn.ModuleDict({'dice': self.losses[self.train_params['params']['switch_to_dice_epoch']], 'logitmse': self.losses[0]})
        self.trainee = SupervisedTrainee(self.segnet, loss=self.losses[0], augmentation=self.synth, metrics=self.metrics)
        self.FTtrainee = FineTunedTrainee.load_from_checkpoint(checkpoint_path=self.last_checkpoint, trainee=self.trainee, loss=self.losses)
        self.checkpoint_callback = ModelCheckpoint(monitor="val_metric_dice", mode="min", every_n_epochs=1, save_last=True, filename='{epoch}-{val_loss:.5f}')
        
        
    def main(self):
        self.pathsOK()
        #os.remove(self.last_checkpoint)
        #self.logParams()

        n_train = int(self.n_volumes // (1 / self.train_params['data']['train_to_val_ratio']))
        n_val = int(self.n_volumes - n_train)

        train_set, val_set = data.random_split(self.dataset, [n_train, n_val], generator=self.seed)

        # Instantiating loaders
        train_loader = data.DataLoader(train_set,  batch_size=self.train_params['params']['train_batch_size'], shuffle=True)
        val_loader = data.DataLoader(val_set, batch_size=self.train_params['params']["val_batch_size"], shuffle=False)

        if self.gpus == 1:
            trainer = Trainer(accelerator='gpu', benchmark=True, devices=1, logger=self.logger, 
                        callbacks=[self.checkpoint_callback], max_epochs=self.train_params['params']['epochs'])
            trainer.fit(self.FTtrainee, train_loader, val_loader)

        elif self.gpus > 1:
            mp.set_start_method('spawn')
            trainer = Trainer(accelerator='gpu', benchmark=True, devices=self.gpus, strategy="ddp",
                              num_nodes=1, callbacks=[self.checkpoint_callback], logger=self.logger, log_every_n_steps=1, max_epochs=self.train_params['params']['epochs'])
            
            n_processes = self.gpus
            self.segnet.share_memory()
            processes = []
            for rank in range(n_processes):
                process = mp.Process(target=trainer.fit(self.FTtrainee, train_loader, val_loader), args=(self.segnet))
                process.start()
                processes.append(process)
            for proc in processes:
                proc.join()
        else:
            print("Couldn't find any GPUs")


    def pathsOK(self):
        if not os.path.exists(f"{self.train_params['paths']['vesselseg_dir']}/models"):
            os.mkdir(f"{self.train_params['paths']['vesselseg_dir']}/models")

        if not os.path.exists(self.version_path):
            os.mkdir(self.version_path)

    def logParams(self):
        json_object = json.dumps(self.train_params["train"], indent=4)
        file = open(f"{self.version_path}/self.train_params.json", 'w+')
        file.write(json_object)
        file.close()

if __name__ == "__main__":
    Train().main()
