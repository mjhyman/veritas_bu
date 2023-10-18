# Standard Imports
import sys

# Environment Settings
sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas")

# Custom Imports
from veritas.data import ImageSynth

if __name__ == "__main__":
    synth = ImageSynth(exp_path="/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0009")
    for i in range(3):
        synth.__getitem__(i, save_nifti=True, make_fig=True, save_fig=True)