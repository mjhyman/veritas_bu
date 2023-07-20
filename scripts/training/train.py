import os
#os.chdir("/autofs/cluster/octdata2/users/epc28/veritas")

import glob
import sys
import nibabel as nib
import torch
from torch.utils.data import Dataset

torch.no_grad()

sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')
import veritas

################ BEGIN TRAIN PARAMETERS ################

train_params = {    
    "train_new": False,
    "paths": {
        "version": 3,
        "data": "/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0008",
        "vesselseg_dir": "/autofs/cluster/octdata2/users/epc28/veritas/output",
        "model": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_3",
        "checkpoint": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_3/checkpoints/epoch=117-val_loss=0.00095.ckpt"
    },
    "data": {
        "n_volumes": 1000,
        "shape": 256,
        "train_to_val_ratio": 0.95
    },
    "model_architecture": {
        "nb_levels": 4,
        "nb_features": [
            16,
            32,
            64,
            128
        ],
        "dropout": 0.1,
        "nb_conv": 2,
        "kernel_size": 3,
        "activation": "ReLU"
    },
    "params": {
        "train_batch_size": 2,
        "val_batch_size": 1,
        "epochs": 1000,
        "switch_to_dice_epoch": 0
    }
}
################ END TRAIN PARAMETERS ################


class SetUp(object):
    
    def __init__(self):
        self.gpus =int(torch.cuda.device_count())

    def model(self, train_new=True):
        sys.path.append("vesselseg")
        from vesselseg.networks import SegNet
        from vesselseg.losses import DiceLoss, LogitMSELoss
        from vesselseg.train import SupervisedTrainee, FineTunedTrainee
        from vesselseg.synth import SynthVesselDataset, SynthVesselImage

        self.segnet = SegNet(3, 1, 1, activation=None, backbone='UNet',kwargs_backbone=(train_params['model_architecture']))
        self.losses = {0: LogitMSELoss(labels=[1]), train_params['params']['switch_to_dice_epoch']: DiceLoss(labels=[1], activation='Sigmoid')}
        self.metrics = torch.nn.ModuleDict({'dice': self.losses[train_params['params']['switch_to_dice_epoch']], 'logitmse': self.losses[0]})
        self.synth = SynthVesselImage()
        self.trainee = SupervisedTrainee(self.segnet, loss=self.losses[0], augmentation=self.synth, metrics=self.metrics)

        if train_new == True:
            self.paths(train_params["paths"]["model"])
            self.FTtrainee = FineTunedTrainee(self.trainee, loss=self.losses)
        elif train_new == False:
            best_checkpoint = train_params['paths']['checkpoint']
            self.FTtrainee = FineTunedTrainee.load_from_checkpoint(checkpoint_path=best_checkpoint, trainee=self.trainee, loss=self.losses)
            self.paths(train_params["paths"]["model"], pattern="events.out")
            self.paths(train_params["paths"]["model"], pattern="last.ckpt")
        return self.FTtrainee, self.segnet
    

    def data(self, device="cuda", synthetic=True):
        '''
        Parameters
        ----------
        synthetic: bool, default True
            sets up the train and val loaders to use real or synthetic data
        '''
        
        seed = torch.Generator().manual_seed(42)
        n_volumes = train_params['data']['n_volumes']

        if synthetic ==  True:
            sys.path.append("vesselseg")
            from vesselseg.synth import SynthVesselDataset
            dataset = SynthVesselDataset(train_params['paths']['data'], subset=slice(n_volumes), device=device)
        
        elif synthetic == False:
            from veritas.data import RealOCTVolumes
            dataset = RealOCTVolumes(device=device, subset=slice(n_volumes))
        
        else:
            print("I don't know what to do with the data!")

        version_path = f"{train_params['paths']['vesselseg_dir']}/models/version_{train_params['paths']['version']}"
        n_train = int(n_volumes // (1 / train_params['data']['train_to_val_ratio']))
        n_val = int(n_volumes - n_train)

        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=seed)

        train_loader = torch.utils.data.DataLoader(train_set,  batch_size=train_params['params']['train_batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=train_params['params']["val_batch_size"], shuffle=False)
        
        return train_loader, val_loader

    
    def trainer(self, trainee, train_loader, val_loader, segnet, device):
        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import TensorBoardLogger
        from pytorch_lightning.callbacks import ModelCheckpoint

        logger = TensorBoardLogger(train_params['paths']['vesselseg_dir'], name="models", version=train_params['paths']['version'])
        checkpoint_callback = ModelCheckpoint(monitor="val_metric_dice", mode="min", every_n_epochs=1, save_last=True, filename='{epoch}-{val_loss:.5f}')

        if self.gpus == 1:
            trainer = Trainer(accelerator=device, benchmark=True, devices=1, logger=logger, 
                        callbacks=[checkpoint_callback], max_epochs=train_params['params']['epochs'])
            trainer.fit(trainee, train_loader, val_loader)

        elif self.gpus > 1:
            import torch.multiprocessing as mp
            print(f"Training on {self.gpus} gpus")
            trainer = Trainer(accelerator=device, benchmark=True, default_root_dir=train_params['paths']['vesselseg_dir'],
                              accumulate_grad_batches=1, devices=self.gpus, strategy="ddp", num_nodes=1, callbacks=[checkpoint_callback],
                              logger=logger, log_every_n_steps=1, max_epochs=train_params['params']['epochs'])
            mp.set_start_method('spawn', force=True)
            n_processes = self.gpus
            segnet.share_memory()
            processes = []
            for rank in range(n_processes):
                process = mp.Process(target=trainer.fit(trainee, train_loader, val_loader), args=(segnet,))
                process.start()
                processes.append(process)
            for proc in processes:
                proc.join()

        return trainer

    def paths(self, path, pattern=None):
        import shutil
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            if pattern is None:
                shutil.rmtree(path)
            elif pattern is not None:
                hits = glob.glob(f"{path}/**/**/**{pattern}**", recursive=True)
                for hit in hits:
                    try:
                        os.remove(hit)
                    except:
                        pass
            else:
                print(f"I don't know what kinda' pattern {pattern} is!")


if __name__ == "__main__":
    trainee, segnet = SetUp().model(train_new=train_params["train_new"])
    train_loader, val_loader = SetUp().data(device="cuda", synthetic=True)
    SetUp().trainer(trainee, train_loader, val_loader, segnet, device="cuda")
    #trainer.fit(trainee, train_loader, val_loader)