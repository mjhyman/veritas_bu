# Custom Imports
from veritas.data import ImageSynth

if __name__ == "__main__":
    synth = ImageSynth(exp_path="/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0008", label_type='skeleton')
    for i in range(10):
        synth.__getitem__(i, save_nifti=True, make_fig=True, save_fig=True)