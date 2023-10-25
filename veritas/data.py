__all__ = [
    'ImageSynth'
    'RealOct',
    'RealOctPatchLoader',
    'RealOctPredict'
]
# Standard imports
import os
import sys
import time
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
# Custom imports
from veritas.utils import Options
from cornucopia.cornucopia import QuantileTransform


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
                 binary_threshold:int=0.5,
                 normalize:bool=False,
                 p_bounds:list[float]=[0, 1],
                 v_bounds:list[float]=[0.05, 0.95],
                 device:str='cuda',
                 pad_:bool=False,
                 padding_method:str='replicate',
                 patch_coords_:bool=False,
                 trainee=None
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

        Notes
        -----
        1. Normalize
        2. Binarize
        3. Convert to dtype
        """
        self.volume=volume
        self.dtype=dtype
        self.patch_size=patch_size
        self.step_size=step_size
        self.binarize=binarize
        self.binary_threshold=binary_threshold
        self.normalize=normalize
        self.p_bounds=p_bounds
        self.v_bounds=v_bounds
        self.device=device
        self.padding_method=padding_method
        self.trainee = trainee
        with torch.no_grad():
            self.volprep()
            if pad_ == True:
                self.reshape()
                self.pad_volume()


    def volprep(self):
        """
        Prepare volume.
        """
        if isinstance(self.volume, str):
            self.volume_name=self.volume.split('/')[-1].strip('.nii')
            self.volume_dir=self.volume.strip('.nii').strip(self.volume_name).strip('/')
            nifti=nib.load(self.volume)
            tensor=torch.tensor(nifti.get_fdata()).to(self.device)
        elif isinstance(self.volume, torch.Tensor):
            tensor = self.volume
            nifti = None
        # Normalize
        if self.normalize == True:
            tensor = tensor.unsqueeze(0)
            tensor = QuantileTransform(
                pmin=self.p_bounds[0], pmax=self.p_bounds[1],
                vmin=self.v_bounds[0], vmax=self.v_bounds[1],
                clip=False
                )(tensor)[0]
        # Binarize
        if self.binarize == True:
            tensor[tensor > self.binary_threshold] = 1
            tensor[tensor <= self.binary_threshold] = 0
        
        tensor = tensor.to(self.device).to(self.dtype)
        self.volume_tensor = tensor.detach()
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
        
    def predict(self):
        self.reshape(4)
        c, x, y, z = self.volume_tensor.shape
        padding = [
            self.patch_size - x,
            0,
            self.patch_size - y,
            0,
            self.patch_size - z,
            0
        ]

        self.volume_tensor = torch.nn.functional.pad(
            input=self.volume_tensor,
            pad = padding,
            mode='replicate'
        )

        prediction = self.trainee(self.volume_tensor.to('cuda').unsqueeze(0))
        prediction = torch.sigmoid(prediction).squeeze().detach().to(self.device)
        prediction = prediction[:x, :y, :z]
        print(prediction.shape)
        print(prediction.max())

        #self.imprint_tensor = torch.zeros(self.volume_tensor.shape)



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
        patch = self.volume_tensor[x_slice, y_slice, z_slice]
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
        ## Needs to go on cuda for prediction
        prediction = self.trainee(patch.to('cuda').unsqueeze(0).unsqueeze(0))
        prediction = torch.sigmoid(prediction).squeeze().detach().to(self.device)
        self.imprint_tensor[coords[0], coords[1], coords[2]] += prediction 
    

    def predict_on_all(self):
        """
        Predict on all patches.
        """
        length = len(self)
        t0 = time.time()
        for idx in range(len(self)):
            self.__getitem__(idx)
            total_elapsed = time.time() - t0
            average_time_per_pred = round(total_elapsed / (idx+1), 2)
            sys.stdout.write(f"\rPrediction {idx + 1}/{length} | {average_time_per_pred} sec/pred | {round(average_time_per_pred * length / 60, 2)} min total pred time")
            sys.stdout.flush()

        # Step size, then number to divide by
        #avg_factors = {256:1, 128:8, 64:64, 32:512, 16:4096}
        patchsize_to_stepsize = self.patch_size // self.step_size
        # for patchsze=256: avg_factor(stepsize=[256,128,64,32]) = [1,8,64,512]
        if self.patch_size == 256:
            avg_factor = 8 ** (patchsize_to_stepsize - 1)
        elif self.patch_size == 128:
            avg_factor = 4 ** (patchsize_to_stepsize - 1)
        elif self.patch_size == 64:
            avg_factor = 2 ** (patchsize_to_stepsize - 1)
        else:
            avg_factor =1
        print(f"\n\n{avg_factor}x Averaging...")
        self.imprint_tensor = self.imprint_tensor / avg_factor
        s = slice(self.patch_size, -self.patch_size)
        self.imprint_tensor = self.imprint_tensor[s, s, s]
    

    def save_prediction(self, dir=None):
        """
        Save prediction volume.

        Parameters
        ----------
        dir : str
            Directory to save volume. If None, it will save volume to same path.
        """
        self.out_dir, self.full_path = Options(self).out_filepath(dir)
        os.makedirs(self.out_dir, exist_ok=True)

        print(f"\nSaving prediction to {self.full_path}...")
        self.imprint_tensor = self.imprint_tensor.cpu().numpy()
        print(self.imprint_tensor.shape)
        print(self.imprint_tensor.max())

        out_nifti = nib.nifti1.Nifti1Image(dataobj=self.imprint_tensor, affine=self.volume_nifti.affine)
        nib.save(out_nifti, self.full_path)