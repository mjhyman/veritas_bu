# Custom Imports
from veritas.models import Unet
from veritas.synth import OctVolSynth

import os
import torch
#torch.no_grad()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    #torch.multiprocessing.set_start_method('spawn')

    # New unet
    #unet = Unet(version_n=1)
    #unet.new(nb_levels=6, nb_features=[32, 64, 128, 256, 512, 1024])
    
    # Load unet (retraining)
    unet = Unet(version_n=2)
    unet.load(type='last')

    unet.train_it(
        data_experiment_number=1,
        augmentation=OctVolSynth(device='cuda'),
        subset=-1,
        train_to_val=0.95,
        epochs=1000,
        batch_size=1,
        loader_device='cuda',
        accumulate_gradient_n_batches=1,
        check_val_every_n_epoch=1
        )