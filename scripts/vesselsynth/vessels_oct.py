__all__ = [
    'VesselSynth'
]
# Standard Imports
import os
import sys
import json
import torch
import nibabel as nib

# Environmet Settings
os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas/vesselsynth")
sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas")

# Custom imports
from veritas.utils import PathTools
from vesselsynth.synth import SynthVesselOCT
from vesselsynth.io import default_affine
from vesselsynth.save_exp import SaveExp
from vesselsynth import backend
backend.jitfields = True


class VesselSynth(object):
    """
    Synthesize 3D vascular network and save as nifti.
    """
    def __init__(self, device:str='cuda',
                 json_param_path:str='scripts/vesselsynth/vesselsynth_params.json'
                 ):
        """
        Parameters
        ----------
        device : 'cuda' or 'cpu' str
            Which device to run computations on
        json_param_path : str
            Location of json file containing parameters
        """
        self.device = device
        self.json_params = json.load(open(json_param_path))   # This is the json file that should be one directory above this one. Defines all variables
        self.shape = self.json_params['shape']                           
        self.n_volumes = self.json_params['n_volumes']
        self.root = self.json_params['output_path']
        self.header = nib.Nifti1Header()
        self.prepOutput(f'{self.root}/vesselsynth_params.json')
        self.backend()
        self.outputShape()


    def synth(self):
        """
        Synthesize a vascular network.
        """
        for n in range(self.n_volumes):
            print(f"Making volume {n:04d}")
            synth_names = ['prob', 'label', "level", "nb_levels",
                         "branch", "skeleton"]
            # Synthesize volumes
            synth_vols = SynthVesselOCT(shape=self.shape, device=self.device)()
            # Save each volume individually
            for i in range(len(synth_names)):
                self.saveVolume(n, synth_names[i], synth_vols[i])   


    def backend(self):
        """
        Check backend for CUDA.
        """
        self.device = torch.device(self.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, using CPU.')
            self.device = 'cpu'


    def outputShape(self):
        if not isinstance(self.shape, list):
            self.shape = [self.shape]
        while len(self.shape) < 3:
            self.shape += self.shape[-1:]


    def saveVolume(self, volume_n:int, volume_name:str, volume:torch.Tensor):
        """
        Save volume as nii.gz.

        Parameters
        ----------
        volume_n : int
            Volume "ID" number
        volume_name : str
            Volume name ['prob', 'label', "level", "nb_levels",\
            "branch", "skeleton"]
        volume : tensor
            Vascular network tensor corresponding with volume_name
        """
        affine = default_affine(volume.shape[-3:])
        nib.save(nib.Nifti1Image(
            volume.squeeze().cpu().numpy(), affine, self.header),
            f'{self.root}/{volume_n:04d}_vessels_{volume_name}.nii.gz')
        
        
    def prepOutput(self, abspath:str):
        """
        Clear files in output dir and log synth parameters to json file.
        
        Parameters
        ---------
        abspath: str
            JSON abspath to log parameters
        """
        PathTools(path=self.root).destroy()
        os.makedirs(self.root, exist_ok=True)
        json_object = json.dumps(self.json_params, indent=4)
        file = open(abspath, 'w')
        file.write(json_object)
        file.close()

        
if __name__ == "__main__":
    VesselSynth().synth()