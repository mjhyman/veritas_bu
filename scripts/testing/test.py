import os
import sys
import json
import math
import numpy as np
from glob import glob
from PIL import Image
import nibabel as nib

import torch
from torch import nn
from torch.utils import data
import torch.multiprocessing as mp

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append("vesselseg")
from vesselseg.networks import SegNet
from vesselseg.losses import DiceLoss, LogitMSELoss
from vesselseg.train import SupervisedTrainee, FineTunedTrainee
from vesselseg.synth import SynthVesselDataset, SynthVesselImage

sys.path.insert(0, "cornucopia")
import cornucopia as cc

os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
torch.set_float32_matmul_precision('medium')

class Test(object):

    def __init__(self):
        self.test_params_path = "scripts/testing/test_params.json"
        self.test_params = json.load(open(self.test_params_path))
        
        # U-Net parameters
        self.version_path = f"{self.test_params['paths']['model']}"
        self.unet_params = json.load(open(f"{self.version_path}/train_params.json"))
        self.segnet = SegNet(3, 1, 1, activation=None, backbone='UNet',kwargs_backbone=(self.unet_params['model_architecture']))
        self.checkpoint_callback = ModelCheckpoint(monitor="val_metric_dice", mode="min", every_n_epochs=1, save_last=True, filename='{epoch}-{val_loss:.5f}')
        self.losses = {0: LogitMSELoss(labels=[1]), 1: DiceLoss(labels=[1], activation='Sigmoid')}
        self.device = "cuda"

        # Data parameters
        self.n_volumes = self.test_params['data']['n_volumes']
        self.dataset = SynthVesselDataset(self.test_params['paths']['data'], subset=slice(self.n_volumes), device=self.device)
        self.seed = torch.Generator().manual_seed(42)
        self.synth = SynthVesselImage()
          
        # Fitting
        self.checkpoint = self.which_checkpoint()
        self.metrics = nn.ModuleDict({'dice': self.losses[1], 'logitmse': self.losses[0]})
        self.trainee = SupervisedTrainee(self.segnet, loss=self.losses[1], augmentation=self.synth, metrics=self.metrics)
        self.FTtrainee = FineTunedTrainee.load_from_checkpoint(checkpoint_path=self.checkpoint, trainee=self.trainee, loss=self.losses)
        self.threshold = 0.25

        # Output
        self.machine_prediction_output_path = f"{self.test_params['paths']['model']}/predictions/niftis"
        self.human_prediction_output_path = f"{self.test_params['paths']['model']}/predictions/tiffs_for_the_humans"

        # Setting up filesystem
        

    def test(self):
        n_train = int(self.n_volumes // (1 / self.test_params['data']['train_to_val_ratio']))
        n_val = int(self.n_volumes - n_train)
        train_set, val_set = data.random_split(self.dataset, [n_train, n_val], generator=self.seed)

        # Instantiating loaders
        val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False)

        self.trainee = self.FTtrainee.trainee
        self.trainee.to(self.device)
        self.trainee.eval()
        
        self.output_dirs()

        with torch.no_grad():
            n = 0
            for batch in val_loader:
                batch = batch.to(self.device)
                img, ref = self.trainee.augmentation(batch)
                seg = self.trainee(img)
                print(type(img))
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


    def which_checkpoint(self):
        '''Determine whether to use the last checkpoint saved in the training of the model, or the best (lowest dice score) in its training.'''
        last_checkpoint_path = [x for x in glob(f"{self.test_params['paths']['model']}/checkpoints/*") if "last" in x]
        best_checkpoint_path = [x for x in glob(f"{self.test_params['paths']['model']}/checkpoints/*") if "val_loss" in x]

        if self.test_params['params']['best_or_last'] == 'best':
            checkpoint_path_used = best_checkpoint_path[0]
            return best_checkpoint_path[0]
        elif self.test_params['params']['best_or_last'] == "last":
            checkpoint_path_used = last_checkpoint_path[0]
        else:
            print("I don't know which checkpoint to use :(")
        return checkpoint_path_used
    

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


    def patch_forward(self, net, img, patch=256, device=None):

        # Calculating half a stride
        mipatch = patch//2
        pad = (mipatch//2, mipatch) * (img.ndim-2)
        out = torch.zeros_like(img)
        print(out.shape)
        img = nn.functional.pad(img, pad)

        if device and torch.device(device).type == 'cuda' and img.device.type == 'cpu':
            img = img.pin_memory()
            out = out.pin_memory()
        n = [int(math.ceil(s/mipatch)) for s in out.shape[2:]]
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    print(i, j, k)
                    block = img[:,:,
                                mipatch*i:mipatch*i+patch,
                                mipatch*j:mipatch*j+patch,
                                mipatch*k:mipatch*k+patch].to(device)
                    pred = net(block).detach()
                    print(block.shape, pred.shape)
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
    
    def realData(self):
        self.output_dirs()

        in_path = "/autofs/cluster/octdata2/users/Chao/caa/caa_17/occipital/process_run1/dBI_vessel/nii_split/I_mosaic_1_1_0.mgz"
        out_path = f'{self.machine_prediction_output_path}/smoothed_prediction.nii.gz'

        tensor = cc.LoadTransform(device='cpu', dtype="float32", to_ras=False)(in_path)[None]

        trainee = self.FTtrainee.trainee
        trainee.to(self.device)
        trainee.eval()

        seg = self.patch_forward(net=trainee, img=tensor, device=self.device)

        seg = torch.sigmoid(seg)
        seg = seg.detach().cpu().numpy()
        affine = nib.load(in_path).affine

        nib.save(nib.Nifti1Image(seg[0, 0], affine), out_path)


    #def makeTiffs(self):
        #nifti_path = "I_mosaic_1_1_0.mgz"
        #prediction_path = f'{self.machine_prediction_output_path}/smoothed_prediction.nii.gz'

        #path = prediction_path
        #out_path = 'output/real_data/prediction_tiffs'

        #dtype = 'float32'
 
        #tensor = cc.LoadTransform(device='cpu', dtype=dtype, to_ras=False)(path)[None]
        #arr = np.array(tensor, dtype=dtype)[0][0]

        #for i in range(self.arr.shape[0]):
        #    img = Image.fromarray(arr[i])
        #    img.save(f"{out_path}/img-{i:04d}.tiff")


#if __name__ == '__main__':
#    Test().test()

#if __name__ == '__main__':
#    Test().realData()
#    Test().makeTiffs()


class RealData(Test):

    def __init__(self):
        super().__init__()
        self.oct_prediction_path = "/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/smoothed_prediction.nii.gz"
        #self.tiff_dir = "output/real_data/prediction_tiffs"
        self.dtype = "float64"
        

    def predict(self):
        self.oct_nifti_path = "/autofs/cluster/octdata2/users/Chao/caa/caa_17/occipital/process_run1/dBI_vessel/nii_split/I_mosaic_1_1_0.mgz"
        self.tensor = cc.LoadTransform(device='cpu', dtype=self.dtype, to_ras=False)(self.oct_nifti_path)[None]
        
        trainee = self.FTtrainee.trainee
        trainee.to(self.device)
        trainee.eval()

        seg = self.patch_forward(net=trainee, img=self.tensor, device=self.device)
        seg = torch.sigmoid(seg)
        seg = seg.detach().cpu().numpy()

        affine = nib.load(self.oct_nifti_path).affine
        nib.save(nib.Nifti1Image(seg[0, 0], affine), self.oct_prediction_path)


    def makeTiffs(self, nifti_path, tiff_dir):
        self.tensor = cc.LoadTransform(device='cpu', dtype=self.dtype, to_ras=False)(nifti_path)[None]
        self.arr = np.array(self.tensor, dtype=self.dtype)[0][0]

        for i in range(self.arr.shape[2]):
            img = Image.fromarray(self.arr[:][:][i]).convert("F")
            img.save(f"{tiff_dir}/img-{i:04d}.tiff")


RealData().makeTiffs(nifti_path="/autofs/cluster/octdata2/users/Chao/caa/caa_17/occipital/process_run1/dBI_vessel/nii_split/I_mosaic_1_1_0.mgz",
                     tiff_dir="/autofs/cluster/octdata2/users/epc28/veritas/output/real_data/tiffs/oct_tiffs")