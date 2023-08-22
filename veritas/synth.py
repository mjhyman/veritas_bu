import torch
from torch import nn
import math as pymath
from torch.utils.data import Dataset

from cornucopia.cornucopia import RandomSmoothLabelMap,\
    RandomGammaNoiseTransform, RandomSlicewiseMulFieldTransform,\
    random


def parenchyma_(T, shape:int=5):
    # Create the label map of int parenchyma but convert to float32 for further computations
    # Add 1 so that we can work with every single pixel (no zeros)
    background_tissue = RandomSmoothLabelMap(random.Fixed(4), shape=shape)(T).to(torch.float32) + 1
    background_tissue = RandomGammaNoiseTransform(sigma=random.Uniform(0.2, 0.4))(background_tissue).to(torch.float32)[0]
    background_tissue = RandomSlicewiseMulFieldTransform()(background_tissue)
    return background_tissue


def vessels_(T, n_groups:int=10, min_i:float=0.25, max_i:float=0.75):
    scaling_tensor = torch.zeros(T.shape).to('cuda')
    vessel_labels = list(sorted(T.unique().tolist()))[1:]
    nb_unique_intensities = random.RandInt(1, n_groups)()
    nb_vessels_per_intensity = int(pymath.ceil(len(vessel_labels) / nb_unique_intensities))

    for int_n in range(nb_unique_intensities):
        intensity = random.Uniform(min_i, max_i)()
        vessel_labels_at_i = vessel_labels[int_n * nb_vessels_per_intensity: (int_n + 1) * nb_vessels_per_intensity]
        for ves_n in vessel_labels_at_i:
            scaling_tensor.masked_fill_(T == ves_n, intensity)
    return scaling_tensor


class OCTSynthVesselImage(nn.Module):

    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.device = 'cuda'
    
    def forward(self, vessel_tensor_labels:torch.Tensor) -> tuple:
        # synthesize the main parenchyma (background tissue)
        parenchyma = parenchyma_(vessel_tensor_labels)
        # Synthesize vessels that are grouped by intensity
        vessels = vessels_(vessel_tensor_labels)
        vessel_texture = parenchyma_(vessel_tensor_labels, shape=10)

        vessel_tensor_labels = vessel_tensor_labels.to(torch.bool)
        vessels = vessels * vessel_texture
        vessels[vessels == 0] = 1

        final_volume = parenchyma * vessels

        return final_volume, vessel_tensor_labels
