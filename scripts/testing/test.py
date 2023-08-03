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
from veritas import utils as vu
#import models

#######################################################################################################
paths = {
    "checkpoint": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_4/checkpoints/last.ckpt",           # string (path)
    #"image_volume": "output/models/version_2/predictions/dylan_data/I_mosaic_1_1_0.nii",                                                        # string (path)
    #"ground_truth": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2/predictions/dylan_data/ground_truth.nii",             # string (path) or None
    #"save_dir": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2/predictions/dylan_data",  
    "image_volume": "/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/I46_Somatosensory_20um_crop.nii",
    "ground_truth": "/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/ground_truth.nii",
    "save_dir": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_4/predictions/caroline_data/prob_maps",                             # string (path)
}

# Don't change this model path
paths["model_path"] = "/".join(paths["checkpoint"].split('/')[:-2])

prediction_settings = {
    "step_size": 64,
    "padding_method": "replicate",
    "device": "cuda",
    "volume_dtype": torch.float16,
    "prediction_dtype": torch.float16
}

options = {
    "threshold": False,
    "fixed_threshold": False,
    "compute_metric": False,
}
#######################################################################################################
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

if __name__ == "__main__":

    t1 = time.time()
    unet = models.UNet(paths["model_path"], paths["checkpoint"])
    oct = vu.OctVolume(path=paths["image_volume"], tile_size=unet.model_params['data']['shape'], step_size=prediction_settings["step_size"])
    oct.predict(unet.trainee)

    y = oct.imprint_tensor
    os.makedirs(paths["save_dir"], exist_ok=True)

    if isinstance(paths["ground_truth"], str):
        ground_truth_tensor, ground_truth_nifti = vu.volprep(paths["ground_truth"], binary=True, device="cuda", dtype=torch.bool)
    else:
        pass

    y, threshold, accuracy = vu.thresholding(prediction_tensor=y,
                                             ground_truth_tensor=ground_truth_tensor,
                                             threshold=False,
                                             auto_threshold=False,
                                             fixed_threshold=0.5)
    
    print(f"\nThreshold: {threshold}")
    print(f"Accuracy: {accuracy}")
    out_filename = f"prediction_stepsz-{prediction_settings['step_size']}_thresh-{threshold}_dice-{accuracy}.nii"
    out_abspath = f"{paths['save_dir']}/{out_filename}"
    print(f"\nSaving to: {out_abspath}")
    
    nifti = nib.nifti1.Nifti1Image(y.cpu().numpy(), affine=oct.volume_nifti.affine, header=oct.volume_nifti.header)
    nib.save(nifti , out_abspath)
        
##########################################################
    t2 = time.time()
    print(f"\nProcess took {round(t2 - t1, 2)} [sec]")