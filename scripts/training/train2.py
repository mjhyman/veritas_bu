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

sys.path.insert(0, "/autofs/cluster/octdata2/users/epc28/veritas/vesselseg/")
from vesselseg.networks import SegNet
from vesselseg.losses import DiceLoss, LogitMSELoss
from vesselseg.train import SupervisedTrainee, FineTunedTrainee
from vesselseg.synth import SynthVesselDataset, SynthVesselImage

sys.path.insert(0, "/autofs/cluster/octdata2/users/epc28/veritas/cornucopia/*")
import cornucopia as cc
from cornucopia.utils.io import loaders


os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
train_params = json.load(open("scripts/training/train_params.json"))


from torch.utils.data import Dataset
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, x_path, y_path, subset=-1, transform=None, target_transform=None):
        self.device = "cpu"
        self.dtype = torch.float32
        self.x_path = x_path[:subset]
        self.y_path = y_path[:subset]

    def __len__(self):
        return len(self.x_path)

    def __getitem__(self, idx):
        image = cc.LoadTransform(dtype=self.dtype, device=self.device)(self.x_path[idx])
        label = cc.LoadTransform(dtype=self.dtype, device=self.device)(self.y_path[idx])
        #image, label = Augmentation(64).run(image, label)
        return image, label


class Train(object):

    def __init__(self):
        self.dtype = np.float32
        self.gpus = int(torch.cuda.device_count())
        self.seed = torch.Generator().manual_seed(42)
        self.n_volumes = 1000
        self.n_train = int(self.n_volumes // (1 / train_params['data']['train_to_val_ratio']))
        self.n_val = int(self.n_volumes - self.n_train)
        self.version_path = f"{train_params['paths']['vesselseg_dir']}/models/version_{train_params['paths']['version']}"

        self.x_path = glob("/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/augmented/x_train/*")
        self.y_path = glob("/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/augmented/y_train/*")
        self.dataset = CustomImageDataset(self.x_path, self.y_path, subset=self.n_volumes)
        self.train_set, self.val_set = data.random_split(self.dataset, [self.n_train, self.n_val], generator=self.seed)

        self.train_loader = data.DataLoader(self.train_set,  batch_size=train_params['params']['train_batch_size'], shuffle=True)
        self.val_loader = data.DataLoader(self.val_set, batch_size=train_params['params']["val_batch_size"], shuffle=False)
        
        self.checkpoint_callback = ModelCheckpoint(monitor="val_metric_dice", mode="min", every_n_epochs=1, save_last=True, filename='{epoch}-{val_loss:.4f}')
        self.logger = TensorBoardLogger(train_params['paths']['vesselseg_dir'], name="models", version=train_params['paths']['version'])
        self.segnet = SegNet(3, 1, 1, activation=None, backbone='UNet',kwargs_backbone=(train_params['model_architecture']))
        self.losses = {0: LogitMSELoss(labels=[1]), train_params['params']['switch_to_dice_epoch']: DiceLoss(labels=[1], activation='Sigmoid')}
        self.metrics = nn.ModuleDict({'dice': self.losses[train_params['params']['switch_to_dice_epoch']], 'logitmse': self.losses[0]})
        self.trainee = SupervisedTrainee(self.segnet, loss=self.losses[0], metrics=self.metrics)
        self.FTtrainee = FineTunedTrainee(self.trainee, loss=self.losses)


    def RealData(self):
        self.pathsOK()
        self.logParams()
        torch.set_float32_matmul_precision('medium')
        if self.gpus == 1:
            trainer = Trainer(accelerator='gpu', benchmark=True, devices=1, logger=self.logger, 
                        callbacks=[self.checkpoint_callback], max_epochs=train_params['params']['epochs'])
            
            trainer.fit(self.FTtrainee, self.train_loader, self.val_loader)
        elif self.gpus > 1:
            trainer = Trainer(accelerator='gpu', benchmark=True, default_root_dir=train_params['paths']['vesselseg_dir'],
                              accumulate_grad_batches=1, devices=self.gpus, strategy="ddp", num_nodes=1, callbacks=[self.checkpoint_callback],
                              logger=self.logger, log_every_n_steps=1, max_epochs=train_params['params']['epochs'])
            mp.set_start_method('spawn', force=True)
            n_processes = self.gpus
            self.segnet.share_memory()
            processes = []
            for rank in range(n_processes):
                process = mp.Process(target=trainer.fit(self.FTtrainee, self.train_loader, self.val_loader), args=(self.segnet,))
                process.start()
                processes.append(process)

            for proc in processes:
                proc.join()
        else:
            print("Couldn't find any GPUs")


    def pathsOK(self):
        if not os.path.exists(f"{train_params['paths']['vesselseg_dir']}/models"):
            os.mkdir(f"{train_params['paths']['vesselseg_dir']}/models")
        if not os.path.exists(self.version_path):
            os.mkdir(self.version_path)
        else:
            for x in glob(f"{self.version_path}/*/*", recursive=True):
                try:
                    os.remove(x)
                except:
                    pass
            #[os.remove(x) for x in glob(f"{self.version_path}/*", recursive=True)]


    def logParams(self):
        json_path = f"{self.version_path}/train_params.json"
        try:
            os.remove(json_path)
        except:
            pass

        try:
            json_object = json.dumps(train_params, indent=4)
            file = open(json_path, 'x+')
            file.write(json_object)
            file.close()
        except:
            pass

if __name__ == "__main__":
    Train().RealData()