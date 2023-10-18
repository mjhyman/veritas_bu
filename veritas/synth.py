__all__ = [
    'OCTSynthVesselImage'
]

# Standard imports
import sys
import torch
from torch import nn
import math as pymath

# Custom packages
sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')
sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas/cornucopia')
from cornucopia.cornucopia.labels import RandomSmoothLabelMap
from cornucopia.cornucopia.noise import RandomGammaNoiseTransform
from cornucopia.cornucopia.intensity import RandomSlicewiseMulFieldTransform
from cornucopia.cornucopia.random import Uniform, Fixed, RandInt


def parenchyma_(vessel_labels_tensor:torch.Tensor, nb_classes:int=4,
                shape:int=5):
    """
    Parameters
    ----------
    vessel_labels_tensor : tensor[int]
        Tensor of vessels with unique ID integer labels
    nb_classes : int
        Number of unique parenchymal "blobs"
    shape : int
        Number of spline control points
    """
    # Create the label map of parenchyma but convert to float32 for further computations
    # Add 1 so that we can work with every single pixel (no zeros)
    parenchyma = RandomSmoothLabelMap(
        nb_classes=Fixed(nb_classes),
        shape=shape
        )(vessel_labels_tensor).to(torch.float32) + 1
    # Applying speckle noise model
    parenchyma = RandomGammaNoiseTransform(
        sigma=Uniform(0.2, 0.4)
        )(parenchyma).to(torch.float32)[0]
    # Applying z-stitch artifact
    parenchyma = RandomSlicewiseMulFieldTransform()(parenchyma)
    return parenchyma


def vessels_(vessel_labels_tensor:torch.Tensor, n_groups:int=10,
             min_i:float=0.25, max_i:float=0.75):
    """
    Parameters
    ----------
    vessel_labels_tensor : tensor[int]
        Tensor of vessels with unique ID integer labels
    n_groups : int
        Number of vessel groups differentiated by intensity
    min_i : float
        Minimum intensity of vessels compared to background
    max_i : float
        Maximum intensity of vessels compared to background
    """
    # Generate an empty tensor that we will fill with vessels and their
    # scaling factors to imprint or "stamp" onto parenchymal volume
    scaling_tensor = torch.zeros(vessel_labels_tensor.shape).to('cuda')
    # Get sorted list of all vessel labels
    vessel_labels = list(sorted(vessel_labels_tensor.unique().tolist()))[1:]
    # Generate the number of unique intensities
    nb_unique_intensities = RandInt(1, n_groups)()
    # Calculate the number of elements (vessels) in each intensity group
    nb_vessels_per_intensity = int(pymath.ceil(len(vessel_labels)
                                               / nb_unique_intensities))
    # Iterate through each vessel group based on their unique intensity
    for int_n in range(nb_unique_intensities):
        # Assign intensity for this group from uniform distro
        intensity = Uniform(min_i, max_i)()
        # Get label ID's of all vessels that will be assigned to this intensity
        vessel_labels_at_i = vessel_labels[int_n * nb_vessels_per_intensity:
                                           (int_n + 1) * nb_vessels_per_intensity]
        # Fill the empty tensor with the vessel scaling factors
        for ves_n in vessel_labels_at_i:
            scaling_tensor.masked_fill_(vessel_labels_tensor == ves_n, intensity)
    return scaling_tensor


class OctVolSynth(nn.Module):

    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.device = 'cuda'
    
    def forward(self, vessel_labels_tensor:torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        vessel_labels_tensor : tensor
            Tensor of vessels with unique ID integer labels
        """
        # synthesize the main parenchyma (background tissue)
        parenchyma = parenchyma_(vessel_labels_tensor)
        # synthesize vessels (grouped by intensity)
        vessels = vessels_(vessel_labels_tensor)
        # Create another parenchyma mask to texturize vessels 
        vessel_texture = parenchyma_(vessel_labels_tensor, shape=10)
        # Texturize vessels!!
        vessels = vessels * vessel_texture
        # Converting label IDs to tensor (we don't need unique IDs anymore,
        # only a binary mask)
        vessel_labels_tensor = vessel_labels_tensor.to(torch.bool)
        # Since it was impossible to get good results with zeros
        vessels[vessels == 0] = 1
        # "stamping" the vessel scaling factor onto the parenchyma volume
        final_volume = parenchyma * vessels
        return final_volume, vessel_labels_tensor
