import os
#os.getcwd()

import time
from glob import glob
import nibabel as nib

import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("cornucopia")
import cornucopia as cc

#sys.path.append("veritas")
#from veritas import models
import models

class OctVolume(Dataset):

    def __init__(self, volume_path, trainee, tile_size, n_averages, subset=-1, transform=None, target_transform=None):
        #torch.set_grad_enabled(False)
        self.volume_path = volume_path
        self.device = 'cuda'
        self.tile_size = tile_size
        self.n_averages = n_averages
        self.volume_dtype = torch.float
        self.imprint_dtype = torch.float32
        self.trainee = trainee

        with torch.no_grad():
            padding = torch.ones(1, 6, dtype=torch.int)
            padding = padding * self.tile_size

            self.volume_nifti = nib.load(self.volume_path)
            self.volume_affine = self.volume_nifti.affine
            self.volume_header = self.volume_nifti.header
            self.volume_tensor = torch.tensor(self.volume_nifti.get_fdata(), device=self.device)

            self.raw_volume_shape = self.volume_tensor.shape
            self.volume_tensor = torch.nn.functional.pad(self.volume_tensor, tuple(*padding), "constant", 0)
            

        #self.volume_raw_shape = self.volume_tensor.shape
        self.x_coords = self.make_patches(self.volume_tensor, 0, self.n_averages)
        self.y_coords = self.make_patches(self.volume_tensor, 1, self.n_averages)
        self.z_coords = self.make_patches(self.volume_tensor, 2, self.n_averages)
        

        self.coordlist = []
        for x in self.x_coords:
            for y in self.y_coords:
                for z in self.z_coords:
                    self.coordlist.append([x, y, z])
        #self.patch_coordinates = self.patch_coords()
        self.imprint_tensor = torch.zeros(self.volume_tensor.shape, dtype=torch.float, device=self.device)

    def __len__(self):
        return len(self.coordlist)


    def __getitem__(self, idx):
        working_coords = self.coordlist[idx]
        x1, x2 = working_coords[0]
        y1, y2 = working_coords[1]
        z1, z2 = working_coords[2]

        tile = self.volume_tensor[slice(x1, x2), slice(y1, y2), slice(z1, z2)].to(self.volume_dtype).detach()
        prediction = self.trainee(tile.unsqueeze(0).unsqueeze(0).to('cuda'))
        prediction = torch.sigmoid(prediction).squeeze().squeeze().detach()

        self.imprint_tensor[slice(x1, x2), slice(y1, y2), slice(z1, z2)] += (prediction / 10)

        return tile, prediction


    def predict(self):
        '''Predict on all patches within 3d volume via getitem function. Normalize resultant imprint and strip padding.'''

        self.volume_tensor = self.volume_tensor / self.volume_tensor.max().item()#cc.QuantileTransform(pmin=0.01, pmax=0.99, vmin=0.01, vmax=0.99, clamp=True)(self.volume_tensor)

        for i in range(self.__len__()):
            self.__getitem__(i)


        # Cropping padding away. Normalizing imprint
        self.imprint_tensor = self.imprint_tensor[self.tile_size: self.tile_size + self.raw_volume_shape[0],
                                                  self.tile_size: self.tile_size + self.raw_volume_shape[1],
                                                  self.tile_size: self.tile_size + self.raw_volume_shape[2]]

        self.imprint_tensor = self.imprint_tensor / self.imprint_tensor.max().item()


        
        self.volume_tensor = self.volume_tensor[self.tile_size: self.tile_size + self.raw_volume_shape[0],
                                                self.tile_size: self.tile_size + self.raw_volume_shape[1],
                                                self.tile_size: self.tile_size + self.raw_volume_shape[2]]


    def make_patches(self, tensor, dim, n_averages):
        step_size = int(self.tile_size // (n_averages**(1/3)))
        n_steps = ((tensor.shape[dim] - 1) // self.tile_size) + 1
        remainder = n_steps * self.tile_size - tensor.shape[dim]
        r1 = np.arange(0, tensor.shape[dim] - self.tile_size, step_size)
        r2 = [x + self.tile_size for x in r1]
        
        if dim == 0:
            torch.nn.functional.pad(self.volume_tensor, (0, 0, 0, 0, 0, remainder))
        elif dim == 1:
            torch.nn.functional.pad(self.volume_tensor, (0, 0, 0, remainder, 0, 0))
        elif dim == 2:
            torch.nn.functional.pad(self.volume_tensor, (0, remainder, 0, 0, 0, 0))
        coord_list = []
        for i in range(len(r1)):
            coord_list.append([r1[i], r2[i]])

        return coord_list


if __name__ == "__main__":
    t1 = time.time()

    #volume_path = "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_0/predictions/real_data/oct_volumes/I_mosaic_0_0_0.mgz"
    #volume_path = "/autofs/cluster/octdata2/users/epc28/veritas/sandbox/tiles/volume-0001.nii"
    volume_path = "/cluster/octdata/users/cmagnain/190312_I46_SomatoSensory/I46_Somatosensory_20um_crop.nii"

    model_path = "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_0"
    unet = models.UNet(model_path)
    oct = OctVolume(volume_path, unet.trainee, tile_size=unet.model_params['data']['shape'], n_averages=10)
    print(oct.x_coords)
    print(oct.y_coords)
    print(oct.z_coords)

    #with torch.no_grad():
    oct.predict()

    x, y = oct.volume_tensor.cpu().numpy(), oct.imprint_tensor.cpu().numpy()
    y[y >= 0.05] = 1
    y[y < 0.05] = 0

    nifti = nib.nifti1.Nifti1Image(y, affine=oct.volume_affine, header=oct.volume_header)
    savedir = f"{model_path}/predictions/caroline_data/"
    os.makedirs(savedir, exist_ok=True)
    nib.save(nifti , f"{model_path}/predictions/caroline_data/prediction_10x-avg.nii")

    #nifti = nib.nifti1.Nifti1Image(x, affine=torch.eye(4))
    #nib.save(nifti , "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_0/predictions/caroline_data/raw_volume.nii")

    t2 = time.time()
    print(f"Process took {round(t2 - t1, 2)} [sec]")
