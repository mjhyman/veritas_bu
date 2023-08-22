# Standard library imports
import os
#os.chdir("/autofs/cluster/octdata2/users/epc28/veritas")
import sys
import time
import numpy as np
import glob as glob
import matplotlib.pyplot as plt

# Related 3rd party imports
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchmetrics.functional import dice, jaccard_index

# Local application/library specific imports
sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas")
#import veritas
import veritas.models as verimod
import veritas.utils as vu
#import cornucopia as cc


#######################################################################################################
paths = {
    "checkpoint": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_7/checkpoints/epoch=457-val_loss=0.00180.ckpt",
    "train_params_json": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_6/train_params.json",
    "image_volume": "/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/I46_Somatosensory_20um_crop.nii",
    "ground_truth": "/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/ground_truth.nii",
    "save_dir": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_7/predictions/caroline_data",                             # string (path)
}

# Don't change this model path
#paths["model_path"] = "/".join(paths["checkpoint"].split('/')[:-2])

prediction_settings = {
    "step_size": 32,
    "padding_method": "replicate",
    "device": "cuda",
    "volume_dtype": torch.float32,
    "prediction_dtype": torch.float32
}

options = {
    "threshold": True,
    "auto_threshold": True,
    "fixed_threshold": False,
}

########################################################################

if __name__ == "__main__":

    t1 = time.time()

    unet = verimod.UNet(
        train_params_json=paths["train_params_json"],
        checkpoint=paths["checkpoint"]
        )
    
    oct = vu.OctVolume(
        path=paths["image_volume"],
        tile_size=unet.model_params['data']['shape'],
        step_size=prediction_settings["step_size"]
        )
    
    # Make the prediction
    oct.predict(unet.trainee)
    y = oct.imprint_tensor
    print(torch.unique(y))

    os.makedirs(paths["save_dir"], exist_ok=True)
    if isinstance(paths["ground_truth"], str):
        ground_truth_tensor, ground_truth_nifti = vu.volprep(
            paths["ground_truth"],
            binary=True,
            device="cuda",
            dtype=torch.bool
            )
    else:
        pass

    y, threshold, accuracy = vu.thresholding(
                                prediction_tensor=y,
                                ground_truth_tensor=ground_truth_tensor,
                                threshold=options['threshold'],
                                auto_threshold=options["auto_threshold"],
                                fixed_threshold=options['fixed_threshold'])
    
    print(f"\nThreshold: {threshold}")
    print(f"Accuracy: {accuracy}")
    out_filename = f"prediction_stepsz-{prediction_settings['step_size']}_thresh-{threshold}_dice-{accuracy}-norm.nii"
    out_abspath = f"{paths['save_dir']}/{out_filename}"
    print(f"\nSaving to: {out_abspath}")
    
    nifti = nib.nifti1.Nifti1Image(
        y.cpu().numpy(),
        affine=oct.volume_nifti.affine,
        header=oct.volume_nifti.header
        )
    nib.save(nifti , out_abspath)
        
########################################################################
    t2 = time.time()
    print(f"\nProcess took {round(t2 - t1, 2)} [sec]")