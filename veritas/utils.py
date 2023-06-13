import torch
import nibabel as nib
from torch.utils.data import Dataset

from glob import glob

x_paths = sorted(glob("/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/augmented/x_train/*"))
y_paths = sorted(glob("/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/augmented/y_train/*"))

class AugmentedVolumes(Dataset):
    def __init__(self, x_paths, y_paths, device="cuda", subset=-1, transform=None, target_transform=None):
        self.device = device
        self.x_paths = x_paths[:subset]
        self.y_paths = y_paths[:subset]

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        with torch.no_grad():
            x = torch.tensor(nib.load(self.x_paths[idx]).get_fdata(), dtype=torch.float, device=self.device)
            y = torch.tensor(nib.load(self.x_paths[idx]).get_fdata(), dtype=torch.float, device=self.device)
            
        return x, y
    

vols = AugmentedVolumes(x_paths, y_paths, subset=9).__getitem__(0)

