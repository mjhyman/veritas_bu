__all__ = [
    'ImageSynth',
]
# Standard Imports
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# Environment Settings
sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas")

# Custom Imports
import nibabel as nib
import cornucopia.cornucopia as cc
from veritas.utils import PathTools
from veritas.synth import OCTSynthVesselImage

imagesynth_params = {
    "path": "/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0009",
    "name": "imagesynth",
    "gamma": [0.99, 1.01],
    "noise": [0.2, 0.6]
}


class ImageSynth(Dataset):
    """
    Synthesize 3D volume from vascular network
    """
    def __init__(self,
                 path_to_labels = imagesynth_params['path'],
                 nifti_dir=f"{imagesynth_params['path']}/volumes/niftis",
                 sample_fig_dir=f"{imagesynth_params['path']}/volumes/samples",
                 device:str="cuda"
                 ):
        """
        Parameters
        ----------
        path_to_labels : str
            Directory where labels (containing unique vessel IDs)
            from vesselsynth are stored.
        nifti_dir : str
            Directory where synthesized nifti volumes will be saved
        sample_fig_dir : str
            Direvtory where sample 2D figures will be saved as .png
        device : {'cuda', 'cpu'}
            Device to run image synthesis on
        """
        self.device = device
        self.imagesynth_params = imagesynth_params
        self.label_paths = sorted(glob.glob(f"{path_to_labels}/*label*"))
        self.nifti_dir = nifti_dir
        self.sample_fig_dir = sample_fig_dir
        PathTools(self.nifti_dir).makeDir()
        PathTools(self.sample_fig_dir).makeDir()


    def __len__(self) -> int:
        return len(self.label_paths)


    def __getitem__(self, idx:int, save_nifti=False, make_fig=False,
                    save_fig=False) -> tuple:
        """
        Parameters
        ----------
        idx : int
            Volume number to synthesize
        save_nifti : int
            Save nifti to self.nifti_dir
        make_fig : int
            Make figure and print it to ipynb output
        save_fig : bool
            Generate and save figure to self.sample_fig_dir
        """
        # Loading nifti and affine
        nifti = nib.load(self.label_paths[idx])
        volume_affine = nifti.affine
        # Loading and processing volume tensor
        volume_tensor = torch.from_numpy(nifti.get_fdata()).to(self.device)
        # Reshaping
        volume_tensor = volume_tensor.squeeze()[None, None]
        # Synthesizing volume
        im, prob = OCTSynthVesselImage()(volume_tensor)
        # Converting image and prob map to numpy. Reshaping
        im = im.detach().cpu().numpy().squeeze().squeeze()
        prob = prob.to(torch.uint8).detach().cpu().numpy().squeeze().squeeze()
        
        if save_nifti == True:
            volume_name = f"volume-{idx:04d}"
            out_path_volume = f'{self.nifti_dir}/{volume_name}.nii.gz'
            out_path_prob = f'{self.nifti_dir}/{volume_name}_MASK.nii.gz'
            print(f"Saving Nifti to: {out_path_volume}")
            nib.save(nib.Nifti1Image(im, affine=volume_affine), out_path_volume)
            nib.save(nib.Nifti1Image(prob, affine=volume_affine), out_path_prob)
        if save_fig == True:
            make_fig = True
        if make_fig == True:
            self.make_fig(im, prob)
        if save_fig == True:
            plt.savefig(f"{self.sample_fig_dir}/{volume_name}.png")

        return im, prob
    

    def make_fig(self, im:np.ndarray, prob:np.ndarray) -> None:
        """
        Make 2D figure (GT, prediction, gt-pred superimposed).
        Print to console.

        Parameters
        ----------
        im : arr[float]
            Volume of x data
        prob: arr[bool] 
            Volume of y data
        """
        plt.figure()
        f, axarr = plt.subplots(1, 3, figsize=(15, 15), constrained_layout=True)
        axarr = axarr.flatten()
        frame = np.random.randint(0, im.shape[0])
        axarr[0].imshow(im[frame], cmap='gray')
        axarr[1].imshow(prob[frame], cmap='gray')
        axarr[2].imshow(im[frame], cmap='gray')
        axarr[2].contour(prob[frame], cmap='magma', alpha=1)
        

if __name__ == "__main__":
    synth = ImageSynth()
    for i in range(3):
        synth.__getitem__(i, save_nifti=True, make_fig=True, save_fig=True)