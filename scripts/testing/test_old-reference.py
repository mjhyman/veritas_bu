import os
import sys
import json
import math
import time
import numpy as np
from glob import glob
from PIL import Image
import nibabel as nib

import torch
from torch import nn
from torch.utils import data
import torch.multiprocessing as mp

sys.path.insert(0, "vesselseg")
from vesselseg.networks import SegNet
from vesselseg.losses import DiceLoss, LogitMSELoss
from vesselseg.train import SupervisedTrainee, FineTunedTrainee
from vesselseg.synth import SynthVesselDataset, SynthVesselImage

sys.path.insert(0, "cornucopia")
import cornucopia as cc

os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
torch.set_float32_matmul_precision('medium')

class Clock():

    def __init__(self):
        self.hr = time.localtime().tm_hour
        self.mi = time.localtime().tm_min
        self.sec = time.localtime().tm_sec

    def now(self):
        return f"{self.hr}:{self.mi}:{self.sec}"


class Predict(object):

    def __init__(self):
        self.device = "cuda"
        self.test_params = json.load(open('scripts/testing/test_params.json'))
        self.model_dir = self.test_params["paths"]["model"]
        self.model_params = json.load(open(f"{self.model_dir}/train_params.json"))

        # U-Net paths
        self.model = SegNet(3, 1, 1, activation=None, backbone="UNet", kwargs_backbone=(self.model_params['model_architecture']))
        self.losses = {0: LogitMSELoss(labels=[1]), 1: DiceLoss(labels=[1], activation='Sigmoid')}
        self.metrics = nn.ModuleDict({'dice': self.losses[1], 'logitmse': self.losses[0]})
        
        self.threshold = 0.10

        self.checkpoint = self.which_checkpoint(self.test_params['params']['best_or_last'])
        self.machine_prediction_output_path = f"{self.test_params['paths']['model']}/predictions/niftis"
        self.human_prediction_output_path = f"{self.test_params['paths']['model']}/predictions/tiffs_for_the_humans"


    def synthData(self):
        self.synth = SynthVesselImage()
        self.trainee = SupervisedTrainee(self.model, loss=self.losses[1], augmentation=self.synth, metrics=self.metrics)
        self.FTtrainee = FineTunedTrainee.load_from_checkpoint(checkpoint_path=self.checkpoint, trainee=self.trainee, loss=self.losses)

        self.trainee = SupervisedTrainee(self.model, loss=self.losses[1], augmentation=self.synth, metrics=self.metrics)
        self.n_volumes = self.test_params['data']['n_volumes']
        self.dataset = SynthVesselDataset(self.test_params['paths']['data'], subset=slice(self.n_volumes), device=self.device)
        self.seed = torch.Generator().manual_seed(42)

        n_train = int(self.n_volumes // (1 / self.test_params['data']['train_to_val_ratio']))
        n_val = int(self.n_volumes - n_train)
        train_set, val_set = data.random_split(self.dataset, [n_train, n_val], generator=self.seed)

        # Instantiating loaders
        val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False)

        self.trainee = self.FTtrainee.trainee
        self.trainee.to(self.device)
        self.trainee.eval()
        
        self.output_dirs()

        print(f"Outputting to {self.machine_prediction_output_path}")

        with torch.no_grad():
            n = 0
            for batch in val_loader:
                batch = batch.to(self.device)
                img, ref = self.trainee.augmentation(batch)
                seg = self.trainee(img)
                #print(type(img))
                seg = torch.sigmoid(seg)
                img = img.detach().cpu().numpy()
                ref = ref.detach().cpu().to(torch.uint8).numpy()
                seg = seg.detach().cpu().numpy()

                for b, (img1, ref1, seg1) in enumerate(zip(img, ref, seg)):
                    nib.save(nib.Nifti1Image(img1[0], None), f'{self.machine_prediction_output_path}/{n:04d}_image.nii.gz')
                    nib.save(nib.Nifti1Image(ref1[0], None), f'{self.machine_prediction_output_path}/{n:04d}_ref.nii.gz')
                    nib.save(nib.Nifti1Image(seg1[0], None), f'{self.machine_prediction_output_path}/{n:04d}_seg.nii.gz')

                    seg1[seg1 <= self.threshold] = 0
                    seg1[seg1 > self.threshold] = 1
                    Image.fromarray(img1[0][img1[0].shape[0] // 2]).save(f"{self.human_prediction_output_path}/{n:04d}_img.tiff")
                    Image.fromarray(seg1[0][seg1[0].shape[0] // 2]).convert("F").save(f"{self.human_prediction_output_path}/{n:04d}_seg.tiff")
                    Image.fromarray(ref1[0][ref1[0].shape[0] // 2]).convert("F").save(f"{self.human_prediction_output_path}/{n:04d}_ref.tiff")
                n += 1



    def realData(self, in_path):
        self.in_dtype = "float32"
        self.out_dtype = np.uint8

        self.losses = {1: DiceLoss(labels=[1], activation='Sigmoid')}
        self.trainee = SupervisedTrainee(self.model, loss=self.losses, metrics=self.metrics)
        self.trainee = FineTunedTrainee.load_from_checkpoint(checkpoint_path=self.checkpoint, trainee=self.trainee, loss=self.losses)
        
        # Getting image name and making output file path
        image_name = in_path.split("/")[-1].split(".")[0]
        self.out_path = f"/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_2/predictions/real_vols/mus_vessel_{image_name}_PREDICTION.nii.gz"

        # Loading data from disk
        print(f"{Clock().now()} Loading data")
        self.tensor = cc.LoadTransform(device='cpu', dtype=self.in_dtype, to_ras=False)(in_path)[None]

        #self.nifti = nib.load(in_path)
        #self.tensor = torch.tensor(self.nifti.get_fdata())
        #self.affine = self.nifti.affine

        # Normalizing data
        print(f"{Clock().now()} Normalizing")
        self.tensor = cc.QuantileTransform(pmin=0, pmax=1, vmin=0.05, vmax=0.95, clamp=False)(self.tensor)
        
        # Setting up network for predictions
        trainee = self.trainee.trainee
        trainee.to(self.device)
        trainee.eval()

        print(f"{Clock().now()} Starting Predictions")
        seg = self.patch_forward(net=trainee, img=self.tensor, device=self.device)
        seg = torch.sigmoid(seg)

        # Thresholding prediction and converting to binary
        print(f"{Clock().now()} Thresholding")
        seg[seg < self.threshold] = 0
        seg[seg >= self.threshold] = 1

        # Converting to numpy int array and taking off gpu
        seg = seg.detach().cpu().numpy().astype(self.out_dtype)

        self.affine = nib.load(in_path).affine
        print(f"{Clock().now()} Saving")
        nib.save(nib.Nifti1Image(seg[0, 0], self.affine), self.out_path)

        #self.makeGif(self.in_path, f"/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/{image_name}.gif")
        #self.makeGif(self.out_path, f"/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/{image_name}_PREDICTION.gif")


    def patch_forward(self, net, img, patch=256, device=None):
        # Calculating half a stride
        mipatch = patch//2
        pad = (mipatch//2, mipatch) * (img.ndim-2)
        out = torch.zeros_like(img)
        #print(out.shape)
        img = nn.functional.pad(img, pad)

        if device and torch.device(device).type == 'cuda' and img.device.type == 'cpu':
            img = img.pin_memory()
            out = out.pin_memory()
            
        n = [int(math.ceil(s/mipatch)) for s in out.shape[2:]]
        
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    #print(i, j, k)
                    block = img[:,:,
                                mipatch*i:mipatch*i+patch,
                                mipatch*j:mipatch*j+patch,
                                mipatch*k:mipatch*k+patch].to(device)
                    pred = net(block).detach()
                    #print(block.shape, pred.shape)

                    outshape = [min(out.shape[-3], mipatch*(i+1)) - mipatch*i,
                                min(out.shape[-2], mipatch*(j+1)) - mipatch*j,
                                min(out.shape[-1], mipatch*(k+1)) - mipatch*k]
                    out[:,:,
                        mipatch*i:mipatch*(i+1),
                        mipatch*j:mipatch*(j+1),
                        mipatch*k:mipatch*(k+1)] = pred[:,:,
                                                        mipatch//2:mipatch//2+outshape[0],
                                                        mipatch//2:mipatch//2+outshape[1],
                                                        mipatch//2:mipatch//2+outshape[2]].to(out)
        return out



    def output_dirs(self):
        '''Make prediction directory and human/machine subdirectories.'''
        if not os.path.exists(f"{self.test_params['paths']['model']}/predictions"):
            os.mkdir(f"{self.test_params['paths']['model']}/predictions")

        if os.path.exists(self.machine_prediction_output_path):
            [os.remove(x) for x in glob(f'{self.machine_prediction_output_path}/*')]
        else:
            os.mkdir(self.machine_prediction_output_path)
        
        if os.path.exists(self.human_prediction_output_path):
            [os.remove(x) for x in glob(f'{self.human_prediction_output_path}/*')]
        else:
            os.mkdir(self.human_prediction_output_path)



    def which_checkpoint(self, which):
        '''Determine whether to use the last checkpoint saved in the training of the model, or the best (lowest dice score) in its training.'''

        checkpoints_available = glob(f"{self.model_dir}/checkpoints/*")

        if which == "best":
            checkpoint_path_used = [x for x in checkpoints_available if "val_loss" in x]
            if len(checkpoint_path_used) == 1:
                checkpoint_path_used = checkpoint_path_used[0]
            else:
                print("More than one available checkpoint! Delete the bad ones")
        elif which == "last":
            checkpoint_path_used = [x for x in checkpoints_available if "last" in x][0]
        else:
            print("I don't know which checkpoint to use :(. Try deleting some :)")

        return checkpoint_path_used



    def makeGif(self, in_path, out_path):
        self.tensor = cc.LoadTransform(device='cpu', dtype=self.dtype, to_ras=False)(in_path)[None]
        self.arr = np.array(self.tensor, dtype=self.dtype)[0][0]

        img_lst = []
        for i in range(self.arr.shape[0]):
            temp_arr = self.arr[i]
            min = temp_arr.min()
            max = temp_arr.max()
            temp_arr = (((temp_arr - min) * 255) // (max - min))
            temp_arr = temp_arr.astype(np.uint8)
            img = Image.fromarray(temp_arr)
            img_lst.append(img)

        img_lst[0].save(out_path, save_all=True, append_images=img_lst[400:900], duration=100, loop=0)


if __name__ == "__main__":
    
    in_path = glob("/autofs/cluster/octdata2/users/Chao/caa/caa_17/occipital/process_run1/mus_vessel/nii_split/I_mosaic_*")[6]
    
    print(f"{Clock().now()} Starting")
    t1 = time.time()
    Predict().realData(in_path)
    t2 = time.time()
    delta_t = round(t2 - t1, 3)

    print(f"Finished in {delta_t / 60} [min]")
    print(f"{Clock().now()} Complete")