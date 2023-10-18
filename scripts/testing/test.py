import sys
import time
sys.path.append('/autofs/cluster/octdata2/users/epc28/veritas')

sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas/cornucopia")
import cornucopia as cc

from veritas.models import NewUnet
from veritas.data import RealOctPredict

import torch.multiprocessing as mp
import torch

data_volumes = [
    '/autofs/cluster/octdata2/users/Chao/caa/caa_17/occipital/process_run1/mus/mus_mean_20um-iso.nii',
    '/autofs/cluster/octdata2/users/Chao/caa/caa_22/processed/20211018/Z_Stitched/mus_mean_20um-iso.nii',
    '/autofs/cluster/octdata2/users/Chao/caa/caa_6/frontal/process_run2/mus/mus_mean_20um-iso.nii',
    '/autofs/cluster/octdata2/users/Chao/caa/caa_6/occipital/process_20220209_run2/mus/mus_mean_20um-iso.nii',
    '/autofs/space/omega_001/users/caa/CAA25_Frontal/Process_caa25_frontal/mus/mus_mean_20um-iso.nii',
    '/autofs/space/omega_001/users/caa/CAA25_Occipital/process_caa25_occipital_run3/mus/mus_mean_20um-iso.nii',
    '/autofs/space/omega_001/users/caa/CAA26_Frontal/Process_caa26_frontal_run2/mus/mus_mean_20um-iso.nii',
    '/autofs/space/omega_001/users/caa/CAA26_Occipital/Process_caa26_occipital/mus/mus_mean_20um-iso.nii',
]



if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    t1 = time.time()
    #volume = data_volumes[5]
    volume = '/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/64x64x64_sub-I38_ses-OCT_sample-BrocaAreaS01_OCT-volume.nii'
    unet_trainee = NewUnet(version_n=8).trainee.trainee
    prediction = RealOctPredict(
        volume=volume,
        trainee=unet_trainee,
        step_size=32,
        device='cuda',
        pad_=True,
        normalize=True
        )
    print('volume loaded...')
    prediction.predict_on_all()
    prediction.save_prediction()

    t2 = time.time()
    print(f"Process took {round((t2-t1)/60, 2)} min")