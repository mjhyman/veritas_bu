# I really have no idea what the best way to sanity check is, so I'm just
# going to import every module that is used :)

from veritas.synth import OctVolSynthDataset

import time
import veritas
import torch

from veritas.models import Unet
from veritas.synth import OctVolSynth

from veritas.synth import VesselSynth


import os
import json
import math as pymath
import nibabel as nib

# Custom Imports
from veritas.utils import PathTools
from vesselsynth.vesselsynth.utils import backend
from vesselsynth.vesselsynth import SaveExp
from vesselsynth.vesselsynth.io import default_affine
from vesselsynth.vesselsynth.synth import SynthVesselOCT
from cornucopia.cornucopia.labels import RandomSmoothLabelMap
from cornucopia.cornucopia.noise import RandomGammaNoiseTransform
from cornucopia.cornucopia import RandomSlicewiseMulFieldTransform
from cornucopia.cornucopia.random import Uniform, Fixed, RandInt


from veritas.utils import PathTools
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset
import numpy as np

import os
import sys
import time
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
# Custom imports
from veritas.utils import Options
from cornucopia.cornucopia import QuantileTransform

# Standard Imports
import torch
from glob import glob
import torch.multiprocessing as mp
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# Custom Imports
from vesselseg.vesselseg import networks, losses, train
from vesselseg.vesselseg.synth import SynthVesselDataset
from veritas.utils import PathTools, JsonTools, Checkpoint

# Standard Imports
import os
import glob
import json
import torch
import shutil
import numpy as np
import math as pymath
from torchmetrics.functional import dice


# Standard Imports
import torch
import numpy as np
import tifffile
# Custom Imports
from veritas.data import RealOct

print("""__   _______ _   _    ___  ______ _____  ______ _____ _   _  _____ 
\ \ / /  _  | | | |  / _ \ | ___ \  ___| |  _  \  _  | \ | ||  ___|
 \ V /| | | | | | | / /_\ \| |_/ / |__   | | | | | | |  \| || |__  
  \ / | | | | | | | |  _  ||    /|  __|  | | | | | | | . ` ||  __| 
  | | \ \_/ / |_| | | | | || |\ \| |___  | |/ /\ \_/ / |\  || |___ 
  \_/  \___/ \___/  \_| |_/\_| \_\____/  |___/  \___/\_| \_/\____/ 
                                                                   
                                                                   """)

print("""  ____    ___  ______      ____    ____    __  __  _      ______   ___       __    __   ___   ____   __  _ 
 /    T  /  _]|      T    |    \  /    T  /  ]|  l/ ]    |      T /   \     |  T__T  T /   \ |    \ |  l/ ]
Y   __j /  [_ |      |    |  o  )Y  o  | /  / |  ' /     |      |Y     Y    |  |  |  |Y     Y|  D  )|  ' / 
|  T  |Y    _]l_j  l_j    |     T|     |/  /  |    \     l_j  l_j|  O  |    |  |  |  ||  O  ||    / |    \ 
|  l_ ||   [_   |  |      |  O  ||  _  /   \_ |     Y      |  |  |     |    l  `  '  !|     ||    \ |     Y
|     ||     T  |  |      |     ||  |  \     ||  .  |      |  |  l     !     \      / l     !|  .  Y|  .  |
l___,_jl_____j  l__j      l_____jl__j__j\____jl__j\_j      l__j   \___/       \_/\_/   \___/ l__j\_jl__j\_j
                                                                                                           """)