import json
import numpy as np
import torch
from torch import nn
from glob import glob

import sys

sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas/vesselseg")
from vesselseg import networks, losses, train

class UNet(object):
        def __init__(self, model_path):
                self.model_path = model_path
                self.device = "cuda"
                self.best_or_last = "best"
                self.model_params = json.load(open(f'{self.model_path}/train_params.json'))
                self.checkpoint = self.which_checkpoint()

                # U-Net paths
                self.model = networks.SegNet(3, 1, 1, activation=None, backbone="UNet", kwargs_backbone=(self.model_params['model_architecture']))
                self.losses = {0: losses.DiceLoss(labels=[1], activation='Sigmoid')}
                self.metrics = nn.ModuleDict({'dice': self.losses[0]})

                self.trainee = train.SupervisedTrainee(self.model, loss=self.losses, metrics=self.metrics)
                self.trainee = train.FineTunedTrainee.load_from_checkpoint(checkpoint_path=self.checkpoint, trainee=self.trainee, loss=self.losses)
                trainee = self.trainee.trainee
                trainee = trainee.to(self.device)
                trainee = trainee.eval()
                self.threshold = 0.10

        def which_checkpoint(self):
                checkpoint_paths = glob(f"{self.model_path}/checkpoints/*")

                last_checkpoint_path = [x for x in checkpoint_paths if "last" in x]
                best_checkpoint_path = [x for x in checkpoint_paths if "val_loss" in x]

                if self.best_or_last == 'best':
                        checkpoint_path_used = best_checkpoint_path[0]
                elif self.best_or_last == "last":
                        checkpoint_path_used = last_checkpoint_path[0]
                else:
                        print("I don't know which checkpoint to use :(")
                return checkpoint_path_used

