# Standard Imports
import sys
# Environment Settings
sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')
# Custom Imports
from veritas.models import NewUnet
from veritas.synth import OctVolSynth

if __name__ == "__main__":
    path = '/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0009'
    unet = NewUnet(version_n=9, augmentation=OctVolSynth())
    unet.train_it(path)