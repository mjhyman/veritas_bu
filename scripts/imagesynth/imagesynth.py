import os
import sys
import json
sys.path.append("vesselseg")

from datetime import datetime

import glob
from vesselseg.synth import SynthVesselImage
import nibabel as nib
import numpy as np
import torch

from PIL import Image


imagesynth_params = {
    "path": "/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0008",
    "name": "imagesynth",
    "gamma": [0.5, 2],
    "noise": [0.2, 0.6]
}


class VesselSynth(object):
    
    def __init__(self):
        #self.imagesynth_params = json.load(open("vesselseg/scripts/imagesynth/imagesynth_params.json"))
        self.imagesynth_params = imagesynth_params
        self.data_dir = self.imagesynth_params["path"]
        self.name = self.imagesynth_params['name']
        self.label_paths = glob.glob(f"{self.data_dir}/*label*")
        self.nifti_dir = f"{self.data_dir}/volumes/{self.name}/niftis"
        self.tiff_dir = f"{self.data_dir}/volumes/{self.name}/tiffs"

    def main(self):
        self.dirsOK()
        label_object = nib.load(self.label_paths[0]) # Change this back to "label_path" in for loop for production ru
        label_tensor = torch.as_tensor(label_object.get_fdata(), dtype=torch.float, device='cuda').squeeze()[None, None]
        label_tensor.to('cuda')
        
        for iteration in range(0, 8):

            im, prob = SynthVesselImage()(label_tensor)
            im = im.detach().cpu().numpy()[0][0] #.squeeze().numpy()
            #prob = prob.int()

            volume_name = f"volume-{0:04d}_augmentation-{iteration:04d}"
            out_path = f'{self.nifti_dir}/{volume_name}.nii.gz'
        
            nib.save(nib.Nifti1Image(im, affine=label_object.affine), out_path)
            #Image.fromarray(im[im.shape[0] // 2]).save(f'{self.tiff_dir}/{volume_name}.tiff')

            print("Saved to: ", out_path)
                #i += 1

    def dirsOK(self):
        if not os.path.exists(self.nifti_dir):
            os.makedirs(self.nifti_dir)
        else:
            pass
            #os.remove(f"{self.nifti_dir}/*")

        if not os.path.exists(self.tiff_dir):
            os.makedirs(self.tiff_dir)
        else:
            [os.remove(x) for x in glob.glob(f"{self.tiff_dir}/*")]


if __name__ == "__main__":
    VesselSynth().main()