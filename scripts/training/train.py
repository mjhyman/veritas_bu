# Custom Imports
from veritas.models import Unet
from veritas.synth import OctVolSynth

import torch
torch.no_grad()

if __name__ == "__main__":
    unet = Unet(version_n=1)
    unet.new(nb_levels=3, nb_features=[32, 64, 128])
    unet.train_it(
        data_experiment_number=1,
        epochs=25,
        batch_size=2,
        augmentation=OctVolSynth())