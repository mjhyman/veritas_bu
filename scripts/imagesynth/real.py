import os
import sys
import json
import math
import numpy as np
from glob import glob
from PIL import Image
import nibabel as nib

import torch
from torch import nn
from torch.utils import data
import torch.multiprocessing as mp

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset

sys.path.insert(0, "cornucopia")
import cornucopia as cc

os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
torch.set_float32_matmul_precision('medium')


class Augmentation(object):

    def __init__(self):
        self.flip = cc.RandomFlipTransform()
        self.rand_patch = cc.RandomPatchTransform(64)
        self.gamma = cc.RandomGammaTransform((0.5, 2))
        self.rescale = cc.QuantileTransform(pmin=0, pmax=1, vmin=0.05, vmax=0.95, clamp=False)
        self.random_translations = cc.RandomAffineTransform(translations=0.25, rotations=0, shears=0, zooms=0, bound='zeros', shared=True)
        self.random_rotations = cc.RandomAffineTransform(translations=0, rotations=45, shears=0, zooms=0, bound='zeros', shared=True)
        self.random_shears = cc.RandomAffineTransform(translations=0, rotations=0, shears=0.125, zooms=0, bound='zeros', shared=True)
        #self.elastic = cc.RandomElasticTransform(shared=True)
    
    def run(self, t1, t2):
        x = t1
        y = t2

        x, y = self.flip(x, y)
        x, y = self.rand_patch(x, y)

        if torch.randint(0, 2, [1]).item() == 0:
            x, y = self.random_translations(x, y)
        
        if torch.randint(0, 2, [1]).item() == 0:
            x, y = self.random_rotations(x, y)

        if torch.randint(0, 2, [1]).item() == 0:
            x, y = self.random_shears(x, y)       

        #if torch.randint(0, 2, [1]).item() == 0:
            #x, y = self.elastic(x, y)

        x = self.rescale(x)
        x = self.gamma(x)
        x = self.rescale(x)

        return x.squeeze(), y.squeeze()


class AcquiredVolumeData(Dataset):
    def __init__(self, x_paths, y_paths, device="cpu", subset=-1, transform=None, target_transform=None):
        self.skip_next_time = False
        self.subset = subset
        self.device = device
        self.x_paths = self.path_handler(x_paths)
        self.y_paths = self.path_handler(y_paths)

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        image = cc.LoadTransform(dtype=torch.float32, device=self.device)(self.x_paths[idx])
        label = cc.LoadTransform(dtype=torch.float32, device=self.device)(self.y_paths[idx])
        #image, label = Augmentation(64).run(image, label)
        return image, label
    
    def path_handler(self, paths):
        if paths.split("/").pop() == "*":
            paths = glob(paths)
        elif paths.split("/").pop().split(".").pop() == "nii":
            paths = [paths]
        else:
            try:
                paths = glob(f"{paths}/*")
            except:
                pass

        if isinstance(paths, list):
            # What to do if x is a list
            if len(paths) == 1:
                return paths
            elif len(paths) > 1: # If paths is a list and bigger than 1
                paths = sorted(paths)
                if len(paths) > self.subset:  # If paths is a list, bigger than 1, and has more elements than the requested subset
                    paths = paths[:self.subset]
                    return paths
                elif len(paths) < self.subset:
                    print("Requested subset is larger than list! Using all list elements")
                    return paths
            else:
                print(paths)
                print("I have no idea what that path is!")


    def augmentAndSave(self, savepath, reps_per_volume):
        if os.path.exists(savepath):
            [os.remove(x) for x in glob(f"{savepath}/*/*")]
        else:
            os.makedirs(f"{savepath}/x_train")
            os.makedirs(f"{savepath}/y_train")

        for i in range(self.__len__()):
            for rep in range(reps_per_volume):
                x, y = Augmentation().run(self.__getitem__(i)[0], self.__getitem__(i)[1])
                
                x = x.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                y[y <=0] = 0
                y[y > 0] = 1

                # Make nifti using identity matrix as affine
                x = nib.Nifti1Image(x, torch.eye(4), dtype=np.float32)
                y = nib.Nifti1Image(y, torch.eye(4), dtype=np.uint8)

                nib.save(x, f"{savepath}/x_train/volume-{i:04d}_augmentation-{rep:04d}")
                nib.save(y, f"{savepath}/y_train/volume-{i:04d}_augmentation-{rep:04d}")


    def look(self):
            idx = torch.randint(0, self.__len__(), (1,)).item()
            image, label = self.__getitem__(idx)
            image = (np.array(image, dtype=np.float16) * 255).astype(np.uint8)
            label = (np.array(label, dtype=np.float16) * 255).astype(np.uint8)
            label[label > 0] = 255

            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(image.squeeze()[0], cmap='gray')
            axarr[1].imshow(label.squeeze()[0], cmap='gray')


def output_dirs(p):
    '''Make prediction directory and human/machine subdirectories.'''
    if os.path.exists(p):
        [os.remove(x) for x in glob(f'{p}/*')]
    else:
        os.mkdir(p)

############ BEGIN AUGMENT AND SAVE ############

x_paths = "/cluster/octdata/users/cmagnain/190312_I46_SomatoSensory/I46_Somatosensory_20um_crop.nii"
y_paths = "/cluster/octdata/users/cmagnain/190312_I46_SomatoSensory/Vessels_20um_crop_v4.nii"

savepath = "/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/nonlinearly-augmented"
if __name__ == "__main__":
    AcquiredVolumeData(x_paths, y_paths).augmentAndSave(savepath, reps_per_volume=20)

############ BEGIN AUGMENT AND SAVE ############