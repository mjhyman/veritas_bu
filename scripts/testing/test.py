import os
os.chdir("/autofs/cluster/octdata2/users/epc28/veritas")

import time
import numpy as np
from glob import glob
import nibabel as nib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchmetrics.functional import dice, jaccard_index

import sys
sys.path.append("cornucopia")
import cornucopia as cc

sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas")
from veritas import models
#import models


#######################################################################################################


paths = {
    "checkpoint": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2/checkpoints/epoch=117-val_loss=0.00095.ckpt",           # string (path)
    #"checkpoint": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_3/checkpoints/epoch=168-val_loss=0.00328.ckpt",
    "image_volume": "output/models/version_2/predictions/dylan_data/I_mosaic_1_1_0.nii",                                                        # string (path)
    "ground_truth": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2/predictions/dylan_data/ground_truth.nii",             # string (path) or None
    "save_dir": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2/predictions/dylan_data/",                                 # string (path)
}

# Don't change this model path
paths["model_path"] = "/".join(paths["checkpoint"].split('/')[:-2])
print(paths['model_path'])


prediction_settings = {
    "step_size": 256,
    "padding_method": "replicate",
    "device": "cuda",
    "volume_dtype": torch.float16,
    "prediction_dtype": torch.float16
}

options = {
    "threshold": True,
    "fixed_threshold": True,
    "compute_metric": True,
}

if options['threshold'] == True:

    if options["fixed_threshold"] == True:
        threshold_settings = {
            "threshold": 0.5
            }
        
    elif options["fixed_threshold"] == False:
        threshold_settings = {
            "start": 0.05,
            "stop": 0.95,
            "step": 0.05,
            }
    else:
        print("I don't know how to threshold!")

if options["compute_metric"] == True:
    metric_type = "dice"                 # string ("dice" or "iou")


#######################################################################################################

class OctVolume(Dataset):

    def __init__(self, volume_path, trainee, tile_size, step_size, device="cpu", subset=-1, transform=None, target_transform=None):
        self.volume_path = volume_path
        self.device = device
        self.tile_size = tile_size
        self.step_size = step_size
        self.volume_dtype = prediction_settings['volume_dtype']
        self.imprint_dtype = prediction_settings["prediction_dtype"]
        self.trainee = trainee
        
        # Get all volume specific things
        with torch.no_grad():
            self.volume_nifti = nib.load(self.volume_path)
            self.volume_affine = self.volume_nifti.affine
            self.volume_header = self.volume_nifti.header
            self.volume_tensor = torch.tensor(self.volume_nifti.get_fdata(), dtype=self.volume_dtype, device=self.device)
            self.volume_tensor = cc.QuantileTransform(pmin=0, pmax=1, vmin=0.01, vmax=0.99, clamp=False)(self.volume_tensor.to(torch.float) + 0.000001).to(self.volume_dtype)
            self.raw_volume_shape = self.volume_tensor.shape    
        # Pad each dimension individually
        self.pad_dimension()
        self.imprint_tensor = torch.zeros(self.volume_tensor.shape, dtype=self.imprint_dtype, device=self.device)
        # Partition volume into overlapping 3d patches
        self.get_frame_coords(step_size=self.step_size)


    def __len__(self):
        return len(self.coordlist)


    def __getitem__(self, idx):
        working_coords = self.coordlist[idx]
        x_slice = slice(*working_coords[0])
        y_slice = slice(*working_coords[1])
        z_slice = slice(*working_coords[2])
        tile = self.volume_tensor[x_slice, y_slice, z_slice].to(self.volume_dtype).detach().to("cuda").to(torch.float)#.to('cpu')
        prediction = self.trainee(tile.unsqueeze(0).unsqueeze(0))#.to('cpu')
        prediction = torch.sigmoid(prediction).squeeze().squeeze().detach()
        self.imprint_tensor[x_slice, y_slice, z_slice] += prediction
        return tile, prediction


    def predict(self):
        '''Predict on all patches within 3d volume via getitem function. Normalize resultant imprint and strip padding.'''
        # Normalizing
        length = self.__len__()
        print("Predicting on", length, 'patches')
        for i in range(length):
            self.__getitem__(i)
            sys.stdout.write(f"\rPrediction {i + 1}/{length}")
            sys.stdout.flush()
        s = slice(self.tile_size, -self.tile_size)
        self.volume_tensor = self.volume_tensor[s, s, s]
        self.imprint_tensor = self.imprint_tensor[s, s, s]


    def pad_dimension(self):
        with torch.no_grad():
            self.volume_tensor = self.volume_tensor.clone().detach().unsqueeze(0)
            if len(self.volume_tensor.shape) == 4:
                padding = torch.ones(1, 6, dtype=torch.int) * self.tile_size
                padding = tuple(*padding)
                self.volume_tensor = torch.nn.functional.pad(self.volume_tensor, padding, prediction_settings["padding_method"]).squeeze()
            else:
                print('Input tensor has shape', self.volume_tensor.shape)


    def get_frame_coords(self, step_size):
        coords = []
        for dim in range(3):
            dim_start_frame = list(np.arange(0, self.volume_tensor.shape[dim] - self.tile_size, step_size))
            # Remove all elements from starting frame list if all they're going to get is padding
            dim_start_frame.remove(0)
            # Remove all elements from starting frame list if all they're going to get is padding
            dim_end_frame = [d + self.tile_size for d in dim_start_frame]
            coords.append(zip(dim_start_frame, dim_end_frame))
            
        for dim in range(len(coords)):
            if dim == 0:
                self.x_coords = [i for i in coords[dim]]
            if dim == 1:
                self.y_coords = [i for i in coords[dim]]
            if dim == 2:
                self.z_coords = [i for i in coords[dim]]
        self.coordlist = []
        try:
            for x in self.x_coords:
                for y in self.y_coords:
                    for z in self.z_coords:
                        self.coordlist.append([x, y, z])
        except:
            for x in self.x_coords:
                for y in self.y_coords:
                    self.coordlist.append([x, y])

                    
def findthresh(prediction, ground_truth, thresh_start, thresh_stop, thresh_step):
    #prediction = prediction / torch.max(prediction)
    threshold_lst = np.arange(thresh_start, thresh_stop, thresh_step)
    lst = []
    for thresh in threshold_lst:
        prediction_temp = prediction.clone()
        prediction_temp[prediction_temp >= thresh] = 1
        prediction_temp[prediction_temp < thresh] = 0
        if metric_type == 'dice':
            metric = dice(prediction_temp, ground_truth, multiclass=False)
        elif metric_type == "iou":
            metric = jaccard_index(preds=prediction_temp, target=ground_truth, task="binary")
        else:
            print("I don't know that metric!")
        lst.append(metric.tolist())
        
    mx = max(lst)
    mx_idx = lst.index(mx)
    return threshold_lst[mx_idx], lst[mx_idx]


if __name__ == "__main__":

    factors = {256 : 0,
               128: 1,
               64: 2}
    
    averaging_factor = 1 / (8 ** factors[prediction_settings["step_size"]])
    print("Averaging by:", averaging_factor)
    t1 = time.time()
    volume_path = paths["image_volume"]
    model_path = paths["model_path"]
    unet = models.UNet(model_path, paths["checkpoint"])
    oct = OctVolume(volume_path, unet.trainee, tile_size=unet.model_params['data']['shape'], step_size=prediction_settings["step_size"], device=prediction_settings["device"])

    with torch.no_grad():
        oct.predict()
        #x = oct.volume_tensor
        y = oct.imprint_tensor 
        y = y * averaging_factor
        y = y / torch.max(y)
        savedir = paths["save_dir"]
        os.makedirs(savedir, exist_ok=True)

        if isinstance(paths["ground_truth"], str):
            ground_truth_nifti = nib.load(paths["ground_truth"])
            ground_truth_tensor = torch.tensor(ground_truth_nifti.get_fdata()).to("cpu")
            ground_truth_tensor[ground_truth_tensor >= 0.5] = 1
            ground_truth_tensor[ground_truth_tensor <= 0.5] = 0
            ground_truth_tensor = ground_truth_tensor.to(torch.bool)
        else:
            pass

##########################################################
        if options["threshold"] == True:
            if options["fixed_threshold"] == True:
                threshold = threshold_settings["threshold"]
            elif options["fixed_threshold"] == False:
                if isinstance(paths["ground_truth"], str):
                    threshold, accuracy = findthresh(y, ground_truth_tensor, threshold_settings["start"], threshold_settings["stop"], threshold_settings["step"])
                else:
                    print("No ground truth!! Can't compute best threshold :(")
            y[y >= threshold] = 1
            y[y < threshold] = 0
            y = y.to(torch.bool)
            human_threshold = round(threshold, 3)
            out_file = f"prediction_stepsz-{prediction_settings['step_size']}_thresh-{human_threshold}.nii"
            if options["compute_metric"] == True:
                accuracy = dice(preds=y.to("cpu"), target=ground_truth_tensor, multiclass=False)
                human_accuracy = round(accuracy.item(), 3)
                out_file = f"prediction_stepsz-{prediction_settings['step_size']}_thresh-{human_threshold}_dice-{human_accuracy}.nii"
        elif options["threshold"] == False:
            out_file = f"prediction_stepsz-{prediction_settings['step_size']}_prob-map.nii"
        else:
            print("IDK what that threshold is!!!")
        out_file_abspath = f"{paths['save_dir']}/{out_file}"
        print(f"\nSaving to: {out_file_abspath}")
        nifti = nib.nifti1.Nifti1Image(y.cpu().numpy(), affine=oct.volume_affine, header=oct.volume_header, dtype=np.uint8)
        nib.save(nifti , out_file_abspath)
        
##########################################################
    t2 = time.time()
    print(f"\nProcess took {round(t2 - t1, 2)} [sec]")