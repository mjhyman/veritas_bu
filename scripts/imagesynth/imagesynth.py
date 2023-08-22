import os
import sys
import json
import glob
import math as pymath
from datetime import datetime

import torch
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset

print(os.getcwd())
#sys.path.append('vesselseg')
sys.path.append('cornucopia/..')
#sys.path.append('vesselsynth')

#import cornucopia
import cornucopia.cornucopia as cc

imagesynth_params = {
    "path": "/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0008",
    "name": "imagesynth",
    "gamma": [0.99, 1.01],
    "noise": [0.2, 0.6]
}

class OCTSynthVesselImage(nn.Module):

    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.device = "cuda"
        self.flip = cc.RandomFlipTransform()
        self.mask = cc.RandomSmoothLabelMap(cc.random.Fixed(4), shape=5)
        self.background_classes = cc.RandomSmoothLabelMap(4, shape=8)
        self.smooth = cc.RandomSmoothTransform()
        self.gmm = cc.RandomGaussianMixtureTransform()
        self.gamma = cc.RandomGammaTransform((0.5, 2))
        self.noise = cc.RandomGammaNoiseTransform(
            cc.random.Fixed(1), sigma=cc.random.Uniform(0.5, 0.6)
            )
        self.norm = cc.QuantileTransform(pmin=0, pmax=1, vmin=0.05, vmax=0.95)
        self.sawtooth = cc.RandomSlicewiseMultFieldTransform(
            shape_through=50, vmax=0.5
            ) 


    def forward(self, vessel_tensor_labels):
        # vessel_tensor_label, vessel_tensor_image,
        # parenchyma_tensor_novessels, parenchyma_tensor_image
        '''vessek_tensor_labels: volume of unique vessel labels.\
        Each vessel has a unique ID on [1, n].\
        Background (parenchymal matter) = 0'''
        # Will create "empty" tensors (e_image, e_binary_mask) that will
        # have outputs imprinted onto them

        vol_shape = vessel_tensor_labels.shape
        c = None
        if isinstance(vessel_tensor_labels, (list, tuple)):
            vessel_tensor_labels, c = vessel_tensor_labels
            print("Is I")   # I'm not sure what this means -epc28

        e_image = torch.empty_like(vessel_tensor_labels, dtype=torch.float32)
        
        if c is not None:
            e_mask = vessel_tensor_labels.new_empty(
                [vol_shape[0], 2, *vol_shape.shape[2:]], dtype=torch.bool
                )
            for frame, (channel, c1) in enumerate(zip(vessel_tensor_labels, c)):
                e_image[frame], e_mask[frame, 0], e_mask[frame, 1] = self.forward1(channel, c1)
        else:
            e_mask = torch.empty_like(vessel_tensor_labels, dtype=torch.bool)
            for frame, channel in enumerate(vessel_tensor_labels):
                e_image[frame], e_mask[frame] = self.forward1(channel)
        return e_image, e_mask


    def forward1(self, vessel_tensor_labels:torch.Tensor, c=None):
        '''
        vessel_tensor_labels:
        Volume of unique vessel labels. Each vessel label has a unique int ID
        '''
        if c is not None:
            vessel_tensor_labels, c = self.flip(vessel_tensor_labels, c)
        else:
            vessel_tensor_labels = self.flip(vessel_tensor_labels)
        m = self.mask(vessel_tensor_labels) > 0
        vessel_tensor_labels *= m
        if c is not None:
            c *= m

        #vtl_cp = vessel_tensor_labels.clone()
        
        # make tensor full of ones for 
        e_vess = torch.ones(vessel_tensor_labels.shape, device=self.device)

        # generate background classes
        image_tensor = self.background_classes(vessel_tensor_labels)
        image_tensor += 1

        # Get all unique vessel ID's into a list that we can iterate over
        vessel_label_list = list(sorted(vessel_tensor_labels.unique().tolist()))[1:]

        # Generate a number of unique intensities from 1 to 5
        nb_unique_intensities = cc.random.RandInt(1, 5)()

        # Calculate how many vessels are per intensity
        nb_vessels_per_intensity = int(pymath.ceil(len(vessel_label_list) / nb_unique_intensities))
        
        for intensity_n in range(nb_unique_intensities):
            # Determine which vessel ID's to color in
            vessel_labels_at_i = vessel_label_list[intensity_n * nb_vessels_per_intensity: (intensity_n + 1) * nb_vessels_per_intensity]
            # Go over each label individually, and color it in randomly
            for idx in vessel_labels_at_i:
                # We should adjust this based on the human eye's just noticeable difference
                # [0.25, 0.75]
                intensity = cc.random.Uniform(0.25, 0.75)()
                e_vess[vessel_tensor_labels == idx] = intensity

        #print("\ne_vess:", torch.unique(e_vess))
        #print("\nBefore transformations:", torch.unique(image_tensor), '\n')

        # Making weighted imprint
        image_tensor = image_tensor.to(torch.float32)
        # Gamma transform [0.5, 1.75]
        image_tensor = cc.RandomGammaTransform((0.99, 1))(image_tensor)
        # Gamma noise model [0.2, 0.4]
        #image_tensor = cc.RandomGammaNoiseTransform(cc.random.Fixed(1), sigma=cc.random.Uniform(0.2, 0.4))(image_tensor)
        # Sawtooth
        #image_tensor = cc.RandomSlicewiseMultFieldTransform(shape_through=50, vmax=0.5)(image_tensor)
        # Normalize
        image_tensor = cc.QuantileTransform(pmin=0, pmax=1, vmin=0.05, vmax=0.95, clamp=True)(image_tensor)
        # Do some post processing
        #image_tensor = image_tensor / torch.max(image_tensor)
        image_tensor = image_tensor * e_vess
        vessel_tensor_labels = vessel_tensor_labels > 0
        vessel_tensor_labels = vessel_tensor_labels.to(torch.int8)
        print(torch.unique(torch.isfinite(vessel_tensor_labels)))
        print(torch.unique(torch.isfinite(image_tensor)))

        #print("\nImage After transformations:", torch.unique(image_tensor))
        #print("\nLabel After transformations:", torch.unique(vessel_tensor_labels))

        vessel_tensor_labels[vessel_tensor_labels == torch.nan] = 0

        if c is not None:
            c = c > 0
            return image_tensor, vessel_tensor_labels, c
        else:
            return image_tensor, vessel_tensor_labels


class ImageSynth(Dataset):

    def __init__(self, device="cuda", subset=-1):
        self.device = device
        self.imagesynth_params = imagesynth_params
        self.label_paths = sorted(glob.glob(f"{self.imagesynth_params['path']}/*label*"))
        self.nifti_dir = f"{self.imagesynth_params['path']}/volumes/niftis"
        self.sample_fig_dir = f"{self.imagesynth_params['path']}/volumes/samples"

        self.checkDirs(self.nifti_dir)
        self.checkDirs(self.sample_fig_dir)


    def __len__(self):
        return len(self.label_paths)


    def __getitem__(self, idx, save_nifti, make_fig=False, save_fig=False):
        print(self.label_paths[idx])
        nifti = nib.load(self.label_paths[idx])
        volume_affine = nifti.affine
        volume_tensor = torch.from_numpy(nifti.get_fdata()).to(self.device)
        volume_tensor = volume_tensor.squeeze()[None, None]
        im, prob = OCTSynthVesselImage()(volume_tensor)
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
                plt.savefig(f"{self.sample_fig_dir}/volume-{idx}")

        return im, prob
    

    def make_fig(self, im, prob):
        plt.figure()
        f, axarr = plt.subplots(1, 3, figsize=(15, 15), constrained_layout=True)
        axarr = axarr.flatten()
        frame = np.random.randint(0, im.shape[0])
        axarr[0].imshow(im[frame], cmap='gray')
        axarr[1].imshow(prob[frame], cmap='gray')
        axarr[2].imshow(im[frame], cmap='gray')
        axarr[2].contour(prob[frame], cmap='magma', alpha=1)


    def checkDirs(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            [os.remove(x) for x in glob.glob(f"{dir_path}/*")]


if __name__ == "__main__":
    for i in range(999):
        im, prob = ImageSynth().__getitem__(i, save_nifti=False, make_fig=False ,save_fig=False)