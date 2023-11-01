import time
import veritas
import torch

#set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:32

if __name__ == "__main__":
    #import os
    #os.environ["TORCH_USE_CUDA_DSA"] = "1"
    #os.environ['CUDA_LAUNCH_BLOCKING']='1'
    #os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:32'
    print(f"CUDA available: {torch.cuda.is_available()}")
    t1 = time.time()
    volume = '/cluster/micro/recon/191124_HK001_CaudalMedulla/Process_new/HK001-CaudalMedulla_20um_averaging.nii'
    unet = veritas.Unet(version_n=1)
    unet.load()
    prediction = veritas.RealOctPredict(
        volume=volume,
        trainee=unet.trainee,
        patch_size=256,
        step_size=128,
        device='cuda',
        pad_=True,
        normalize=True
        )
    print('volume loaded...')
    prediction.predict_on_all()
    prediction.save_prediction(f"{unet.version_path}/predictions")
    t2 = time.time()
    print(f"Process took {round((t2-t1)/60, 2)} min")