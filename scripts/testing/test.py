# Standard Imports
import os
import sys
import time
import torch
import glob as glob
import nibabel as nib

# Environment Settings
sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas")

# Custom Imports
from veritas.utils import OctVolume, Thresholding, VolumeUtils
from veritas.models import UNet


########################################################################
paths = {
    "checkpoint": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_8/checkpoints/epoch=135-val_loss=0.00054.ckpt",
    "train_params_json": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_6/train_params.json",
    "image_volume": "/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/I46_Somatosensory_20um_crop.nii",
    #"image_volume": "/autofs/cluster/octdata2/users/epc28/veritas/data/I_mosaic_0_0_0.nii",
    # None or file path
    "ground_truth": "/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/ground_truth.nii",
    "save_dir": "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_8/predictions/caroline_data",                             # string (path)
}

# Don't change this model path
#paths["model_path"] = "/".join(paths["checkpoint"].split('/')[:-2])

prediction_settings = {
    "step_size": 64,
    "padding_method": "replicate",
    "device": "cuda",
}

threshold = 'auto'

########################################################################

if __name__ == "__main__":
    t1 = time.time()
    unet = UNet(
        train_params_json=paths["train_params_json"],
        checkpoint=paths["checkpoint"]
        )
    oct = OctVolume(
        path=paths["image_volume"],
        patch_size=unet.model_params['data']['shape'],
        step_size=prediction_settings["step_size"]
        )    
    # Make the prediction
    oct.predict(unet.trainee)
    y = oct.imprint_tensor

    os.makedirs(paths["save_dir"], exist_ok=True)
    if isinstance(paths['ground_truth'], str):
        gt = VolumeUtils(paths['ground_truth'])
        gt.volprep(binary=True, dtype=torch.bool)
        ground_truth_tensor = gt.volume_tensor
        ground_truth_nifti = gt.volume_nifti
    elif paths["ground_truth"] is None:
        print('\nNo ground truth')
        ground_truth_tensor = None
    else:
        print("I don't know what that ground truth is!!!")
        exit(0)

    y, threshold, accuracy = Thresholding(
        prediction_tensor=y,
        ground_truth_tensor=ground_truth_tensor,
        threshold=threshold,
        compute_accuracy=True
        ).apply()
    
    print(f"\nThreshold: {threshold}")
    print(f"Accuracy: {accuracy}")
    out_filename = paths['image_volume'].split('/')[-1].strip('.nii')
    out_filename = f"{out_filename}_prediction_stepsz-{prediction_settings['step_size']}_thresh-{threshold}_dice-{accuracy}.nii"
    out_abspath = f"{paths['save_dir']}/{out_filename}"
    print(f"\nSaving to: {out_abspath}")
    
    nifti = nib.nifti1.Nifti1Image(
        y.cpu().numpy(),
        affine=oct.volume_nifti.affine,
        header=oct.volume_nifti.header
        )
    nib.save(nifti, out_abspath)
        
########################################################################
    t2 = time.time()
    print(f"\nProcess took {round(t2 - t1, 2)} [sec]")