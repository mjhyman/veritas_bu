# Standard Imports
import sys

# Environment Settings
sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')

# Custom Imports
from veritas.models import UnetTrain
from veritas.synth import OctVolSynth

if __name__ == "__main__":
    path = '/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0008'
    unet = UnetTrain(version_n=9, augmentation=OctVolSynth())
    unet.train_it(path)