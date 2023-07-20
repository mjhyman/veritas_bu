import sys
import torch
import numpy as np
import nibabel as nib

sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas/cornucopia")
#import cornucopia.cornucopia as cc
import cornucopia as cc

def pad_dimension(volume_tensor:torch.Tensor, tile_size:int=256, padding_method:str="replicate") -> torch.Tensor:
    '''Pads all 3 dimensions of a volume with a specific mode according to patch size of unet'''
    with torch.no_grad():
        volume_tensor = volume_tensor.clone().detach().unsqueeze(0)
        if len(volume_tensor.shape) == 4:
            padding = torch.ones(1, 6, dtype=torch.int) * tile_size
            padding = tuple(*padding)
            volume_tensor = torch.nn.functional.pad(volume_tensor, padding, padding_method).squeeze()
            return volume_tensor
        else:
            print('Input tensor has shape', volume_tensor.shape)

##### DEBUGGING FOR pad_dimension() #####
#t = torch.ones(400, 400, 400)
#t = pad_dimension(t, 256, 0, "reflect")
#print(t.shape)


def get_patch_coords(tensor:torch.Tensor, tile_size:int=256, step_size:int=256) -> tuple:
    '''Gets coordinates to all necessary sliding window points'''
    coords = []
    complete_patch_coords = []
    for dim in range(len(tensor.shape)):
        frame_start = np.arange(0, tensor.shape[dim] - tile_size, step_size)[1:]
        frame_end = [d + tile_size for d in frame_start]
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

##### DEBUGGING FOR get_patch_coords() #####
#tensor = torch.ones(912, 1656, 912)
#coords = get_patch_coords(tensor, 256, 64)
#coords

def volprep(path:str, binary: bool=False, device: str="cpu", dtype: torch.dtype=torch.float, vmin:float=0.01, vmax:float=0.99, clamp:bool=False) -> torch.Tensor:
    '''load, normalize, binarize, device. (returns tensor and nifti header/file info)'''
    with torch.no_grad():
        if path.split(".")[-1] == "nii":
            nifti = nib.load(path)
            tensor = torch.tensor(nifti.get_fdata())
            if binary == True:
                tensor[tensor >= 1] = 1
                tensor[tensor <= 0] = 0
            elif binary == False:
                tensor = cc.QuantileTransform(pmin=0, pmax=1, vmin=vmin, vmax=vmax, clamp=clamp)(tensor)
            tensor = tensor.to(device).to(dtype)
            return tensor, nifti
        else:
            print("Path is not a nifti. Convert to nifti using mri_convert -ot nii <inpath> <outpath>")
            sys.exit(0)

#path = "output/models/version_2/predictions/dylan_data/I_mosaic_1_1_0.nii"
#path = "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2/predictions/dylan_data/ground_truth.nii"
#tensor, nifti = volprep(path, binary=True, device="cuda", dtype=torch.float16)

from torch.utils.data import Dataset

class OctVolume(Dataset):

    def __init__(self, path:str):
        self.tile_size = 256
        self.step_size = 128
        self.volume_tensor, self.volume_nifti = volprep(path)
        self.volume_tensor = pad_dimension(self.volume_tensor)
        self.complete_patch_coords = get_patch_coords(self.volume_tensor, step_size=self.step_size)
        self.imprint_tensor = torch.zeros(self.volume_tensor.shape, dtype=torch.float16, device='cuda')

    def __len__(self):
        return len(self.complete_patch_coords)
    
    def __getitem__(self, idx:int, prediction:bool=False, trainee=None):
        working_coords = self.complete_patch_coords[idx]
        x_slice = slice(*working_coords[0])
        y_slice = slice(*working_coords[1])
        z_slice = slice(*working_coords[2])
        tile = self.volume_tensor[x_slice, y_slice, z_slice].detach().to("cuda")
        if prediction == True:
            prediction = trainee(tile.unsqueeze(0).unsqueeze(0))
            prediction = torch.sigmoid(prediction).squeeze().squeeze().detach()
            self.imprint_tensor[x_slice, y_slice, z_slice] += prediction
            return tile, prediction
        elif prediction == False:
            return tile

    def predict(self, trainee):
        '''Predict on all patches within 3d volume via getitem function. Normalize resultant imprint and strip padding.'''
        length = self.__len__()
        print("Predicting on", length, 'patches')
        for i in range(length):
            self.__getitem__(i, prediction=True, trainee=trainee)
            sys.stdout.write(f"\rPrediction {i + 1}/{length}")
            sys.stdout.flush()
        s = slice(self.tile_size, -self.tile_size)
        # Removing Padding
        self.volume_tensor = self.volume_tensor[s, s, s]
        self.imprint_tensor = self.imprint_tensor[s, s, s]


if __name__ == "__main__":
    sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas")
    from veritas import models

    factors = {256 : 0,
            128: 1,
            64: 2}

    path = "output/models/version_2/predictions/dylan_data/I_mosaic_1_1_0.nii"
    unet = models.UNet("/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2", "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2/checkpoints/epoch=117-val_loss=0.00095.ckpt")
    oct = OctVolume(path)
    oct.predict(unet.trainee)
    averaging_factor = 1 / (8 ** factors[oct.step_size])
    print(averaging_factor)
    y = oct.imprint_tensor * averaging_factor
    y = y / torch.max(y)
    nifti = nib.nifti1.Nifti1Image(y.cpu().numpy(), affine=oct.volume_nifti.affine, header=oct.volume_nifti.header)
    nib.save(nifti, "/autofs/cluster/octdata2/users/epc28/veritas/test-128.nii")

    #oct.__getitem__(0, prediction=True, trainee=unet.trainee)