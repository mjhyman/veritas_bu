import sys
import torch
import numpy as np
#import math as pymath
import nibabel as nib
#import matplotlib.pyplot as plt
#from torch.utils.data import Dataset
#from torchmetrics.functional import dice

# Need tools to:
## Create new json param folders, import data better, delete dirs

sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas")
import cornucopia.cornucopia as cc

class VolumeUtils(object):
    """
    Base class for volume operations
    """
    def __init__(self,
                 volume:{torch.Tensor, 'path'},
                 patch_size:int=256,
                 step_size:int=256
                 ):
        """
        Parameters
        ----------
        volume: {torch.Tensor, 'path'} |\
            Tensor of entire volume or path to tensor.
        volume_tensor: torch.Tensor |\
            Tensor of entire volume.
        patch_size: int |\
            Size of patch with which to partition volume into.
        step_size: int |\
            Size of step between adjacent patch origin.
        """
        self.volume=volume
        self.volume_tensor, self.nifti = self.volprep()
        self.patch_size=patch_size
        self.step_size=step_size
        self.volume_tensor = self.pad_volume()
        self.complete_patch_coords = self.patch_coords()
        self.volume_shape = self.volume_tensor.shape

    def volprep(self,
                binary:bool=False,
                dtype:torch.dtype=torch.float32,
                pmin:float=0,
                pmax:float=1,
                vmin:float=0.05,
                vmax:float=0.95,
                device:str="cuda"
                ) -> tuple:
        """
        Prepare volume.

        Parameters
        ----------
        path : str
            Path to (nifti)
        binary : bool
            Binarize volume (values >= 1 set to 1, values < 1 set to 0)
        dtype : torch.dtype
            Load volume as this data type
        pmin : float
            Lower end quantile for histogram adjustment
        pmax : float
            Higher end quantile for histogram adjustment
        vmin : float
            Shift pmin to this quantile
        vmax : float
            Shift pmax to this quantile
        clip : bool
            Clip histogram that falls outside vmin and vmax
        device : device 
        """
        if isinstance(self.volume, str):
            nifti=nib.load(self.volume)
            tensor=torch.tensor(nifti.get_fdata())
        elif isinstance(self.volume, torch.Tensor):
            tensor = self.volume
            nifti = None

        if binary == True:
            tensor[tensor >= 1] = 1
            tensor[tensor < 1] = 0
        elif binary == False:
            tensor = tensor.unsqueeze(0)
            tensor = cc.QuantileTransform(pmin=pmin, pmax=pmax, vmin=vmin, vmax=vmax, clip=False)(tensor)
            tensor = tensor[0]
        tensor = tensor.to(device).to(dtype)
        return tensor, nifti


    def reshape(self, volume_tensor:torch.Tensor, shape:int=4) -> torch.Tensor:
        """
        Ensure tensor has proper shape

        Parameters
        ----------
        shape: int |\
            Shape that tensor needs to be.
        """
        if len(volume_tensor.shape) < shape:
            volume_tensor = volume_tensor.unsqueeze(0)
            return volume_tensor
        elif len(volume_tensor.shape) == shape:
            return volume_tensor
        else:
            print("Check the shape of your volume tensor plz.")
            exit(0)


    def pad_volume(self, padding_method='replicate'):
        """
        Pad all dimensions of 3D volume according to patch size.

        Parameters
        ----------
        pading_method: {'replicate', 'reflect', 'constant'} |\
            How to pad volume.
        """
        self.padding_method = padding_method
        with torch.no_grad():
            volume_tensor = self.volume_tensor.clone().detach()
            volume_tensor = self.reshape(volume_tensor, 4)
            padding = torch.ones(1, 6, dtype=torch.int) * self.patch_size
            padding = tuple(*padding)
            volume_tensor = torch.nn.functional.pad(
                input=volume_tensor,
                pad=padding,
                mode=padding_method
            ).squeeze()
            return volume_tensor


    def patch_coords(self):
        """
        Compute coords for all patches.
        """
        coords = []
        complete_patch_coords = []
        vol_shape = self.volume_tensor.shape
        for dim in range(len(vol_shape)):
            frame_start = np.arange(
                0, vol_shape[dim] - self.patch_size, self.step_size
                )[1:]
            frame_end = [d + self.patch_size for d in frame_start]
            coords.append(list(zip(frame_start, frame_end)))
        if len(coords) == 3:
            for x in coords[0]:
                for y in coords[1]:
                    for z in coords[2]:
                        complete_patch_coords.append([x, y, z])
        elif len(coords) == 2:
            for x in coords[0]:
                for y in coords[1]:
                    complete_patch_coords.append([x, y])  
        return complete_patch_coords


vol = torch.ones([1, 20, 20, 20])
new_vol = VolumeUtils(volume=vol)
print(new_vol.complete_patch_coords)