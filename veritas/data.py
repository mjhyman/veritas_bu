__all__ = [
    'RealOct',
    'RealOctPatchLoader',
    'RealOctPredict'
]

import sys
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas/cornucopia")
import cornucopia as cc

sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')
from veritas.models import UNet
from veritas.utils import Options


class RealOct(object):
    """
    Base class for real OCT volumetric data.
    """
    def __init__(self,
                 volume:{torch.Tensor, "path"},
                 dtype:torch.dtype=torch.float32,
                 patch_size:int=256,
                 step_size:int=256,
                 binarize:bool=False,
                 normalize:bool=False,
                 p_bounds:list[float]=[0, 1],
                 v_bounds:list[float]=[0.05, 0.95],
                 device:str='cuda',
                 pad_:bool=True,
                 padding_method:str='replicate',
                 patch_coords_:bool=True
                 ):

        """
        Parameters
        ----------
        volume: {torch.Tensor, 'path'}
            Tensor of entire volume or path to nifti.
        dtype: torch.dtype
            Data type to load volume as.
        patch_size: int
            Size of patch with which to partition volume into.
        step_size: int {256, 128, 64, 32, 16}
            Size of step between adjacent patch origin.
        binarize: bool
            Whether to binarize volume.
        normalize: bool
            Whether to normalize volume.
        p_bounds: list[float]
            Bounds for normalization percentile (only if normalize=True).
        v_bounds: list[float]
            Bounds for histogram after normalization (only if normalize=True).
        device: {'cuda', 'cpu'}
            Device to load volume onto.
        padding_method: {'replicate', 'reflect', 'constant'}
            How to pad volume.

        Attributes
        ----------
        volume_nifti
            Nifti represnetation of volumetric data.
        """
        self.volume=volume
        self.volume_name=volume.split('/')[-1].strip('.nii')
        self.volume_dir=volume.strip('.nii').strip(self.volume_name).strip('/')
        self.dtype=dtype
        self.patch_size=patch_size
        self.step_size=step_size
        self.binarize=binarize
        self.normalize=normalize
        self.p_bounds=p_bounds
        self.v_bounds=v_bounds
        self.device=device
        self.padding_method=padding_method
        with torch.no_grad():
            self.volprep()
            self.reshape()
            self.pad_volume()


    def volprep(self):
        """
        Prepare volume.
        """
        if isinstance(self.volume, str):
            nifti=nib.load(self.volume)
            tensor=torch.tensor(nifti.get_fdata())
            #self.volume_name = self.volume.split['/']
        elif isinstance(self.volume, torch.Tensor):
            tensor = self.volume
            nifti = None
        if self.binarize == True:
            tensor[tensor >= 1] = 1
            tensor[tensor < 1] = 0
        elif self.binarize == False:
            if self.normalize == True:
                tensor = tensor.unsqueeze(0)
                tensor = cc.QuantileTransform(
                    pmin=self.p_bounds[0], pmax=self.p_bounds[1],
                    vmin=self.v_bounds[0], vmax=self.v_bounds[1],
                    clip=False
                    )(tensor)[0]
        tensor = tensor.to(self.device).to(self.dtype)
        self.volume_tensor = tensor
        self.volume_nifti = nifti


    def reshape(self, shape:int=4) -> torch.Tensor:
        """
        Ensure tensor has proper shape

        Parameters
        ----------
        shape: int
            Shape that tensor needs to be.
        """
        if len(self.volume_tensor.shape) < shape:
            self.volume_tensor = self.volume_tensor.unsqueeze(0)
        elif len(self.volume_tensor.shape) == shape:
            pass    
        else:
            print("Check the shape of your volume tensor plz.")
            exit(0)


    def pad_volume(self):
        """
        Pad all dimensions of 3D volume and update volume_tensor.
        """
        self.reshape(4)
        padding = torch.ones(1, 6, dtype=torch.int) * self.patch_size
        padding = tuple(*padding)
        self.volume_tensor = torch.nn.functional.pad(
            input=self.volume_tensor,
            pad=padding,
            mode=self.padding_method
            ).squeeze()


class RealOctPatchLoader(RealOct, Dataset):
    """
    Load volumetric patches from real oct volume data.

    Example
    -------
    path = "/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/I46_Somatosensory_20um_crop.nii"
    vol = RealOctPatchLoader(volume=path, step_size=64)
    print(vol[0])
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_coords()


    def __len__(self):
        return len(self.complete_patch_coords)


    def __getitem__(self, idx:int):
        working_patch_coords = self.complete_patch_coords[idx]
        # Generating slices for easy handling
        x_slice = slice(*working_patch_coords[0])
        y_slice = slice(*working_patch_coords[1])
        z_slice = slice(*working_patch_coords[2])
        # Loading patch via coords and detaching from tracking
        patch = self.volume_tensor[x_slice, y_slice, z_slice].detach().to(self.device)
        coords = [x_slice, y_slice, z_slice]
        return patch, coords
        

    def patch_coords(self):
        """
        Compute coords for all patches.

        Attributes
        -------
        complete_patch_coords
            List of all patch coordinates
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

        self.complete_patch_coords = complete_patch_coords


class RealOctPredict(RealOctPatchLoader, Dataset):
    """
    Class for whole OCT volume prediction.

    Parameters
    ----------
    trainee
        ML trainee.

    Attributes
    ----------
    imprint_tensor
        Tensor containing imprint of prediction that gets updated.

    Example
    -------
    unet = veritas.models.UNet(train_params_json, checkpoint)
    vol = RealOctPredict(volume, step_size, trainee=unet.trainee)
    vol.predict_on_all()
    vol.imprint_tensor
    """
    def __init__(self, trainee=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainee=trainee
        self.imprint_tensor=torch.zeros(
            self.volume_tensor.shape, device=self.device, dtype=self.dtype
            )


    def __getitem__(self, idx:int):
        """
        Predict on a single patch.

        Parameters
        ----------
        idx : int
            Patch ID number to predict on. Updates self.imprint_tensor.
        """     
        patch, coords = super().__getitem__(idx)
        prediction = self.trainee(patch.unsqueeze(0).unsqueeze(0))
        prediction = torch.sigmoid(prediction).squeeze().detach()
        self.imprint_tensor[coords[0], coords[1], coords[2]] += prediction
    

    def predict_on_all(self):
        """
        Predict on all patches.
        """
        length = len(self)
        for idx in range(len(self)):
            self.__getitem__(idx)
            sys.stdout.write(f"\rPrediction {idx + 1}/{length}")
            sys.stdout.flush()

        # Step size, then number to divide by
        avg_factors = {256:1, 128:8, 64:64, 32:512, 16:4096}
        self.imprint_tensor = self.imprint_tensor / avg_factors[self.step_size]
    

    def save_prediction(self):
        """
        Save prediction volume.
        """
        self.out_fname = Options(self).out_filepath()
        print('\n', f"Saving prediction to {self.out_fname}")

if __name__ == "__main__":
    small_vol = "/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/I46_Somatosensory_20um_crop.nii"
    big_vol = "/autofs/cluster/octdata2/users/epc28/veritas/data/I_mosaic_0_0_0.nii"
    unet_params = "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_6/train_params.json"
    checkpoint = "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_8/checkpoints/epoch=135-val_loss=0.00054.ckpt"
    unet = UNet(train_params_json=unet_params, checkpoint=checkpoint)
    vol = RealOctPredict(volume=small_vol, step_size=256, trainee=unet.trainee)
    vol.predict_on_all()
    vol.save_prediction()