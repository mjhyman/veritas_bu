# Custom Imports
from veritas.synth import OctVolSynthDataset

if __name__ == "__main__":
    synth = OctVolSynthDataset(exp_path="/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0009", label_type='label')
    for i in range(5):
        synth.__getitem__(i, save_nifti=True, make_fig=True, save_fig=True)