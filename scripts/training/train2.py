# Custom Imports
from veritas.data import ImageSynth
from veritas.models import UnetTrain
from veritas.synth import OctVolSynth
from vesselseg.vesselseg.synth import SynthVesselDataset

path = '/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0008'

#image_synth = ImageSynth(exp_path=path, label_type='skeleton')
synth_dataset = SynthVesselDataset(inputs=f'{path}/*label*')
print(synth_dataset)