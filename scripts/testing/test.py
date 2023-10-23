import time
import veritas
import torch

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    t1 = time.time()
    volume = '/autofs/cluster/octdata2/users/epc28/veritas/data/caroline_data/I46_Somatosensory_20um_crop.nii'
    unet = veritas.Unet(version_n=8)
    unet.load()
    prediction = veritas.RealOctPredict(
        volume=volume,
        trainee=unet.trainee,
        patch_size=256,
        step_size=64,
        device='cuda',
        pad_=True,
        normalize=True
        )
    print('volume loaded...')
    prediction.predict_on_all()
    prediction.save_prediction()

    t2 = time.time()
    print(f"Process took {round((t2-t1)/60, 2)} min")