__all__ = [
    'VolumeUtils',
    'Thresholding',
    'PathTools',
    'JsonTools'
]
import os
import sys
import json
import torch
import shutil
import numpy as np
import math as pymath
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchmetrics.functional import dice

sys.path.append("/autofs/cluster/octdata2/users/epc28/veritas/cornucopia")
import cornucopia as cc

class VolumeUtils(object):
    """
    Base class for volume operations
    """
    def __init__(self,
                 volume:{torch.Tensor, 'path'},
                 patch_size:int=256,
                 step_size:int=256,
                 pad_:bool=True,
                 patch_coords_:bool=True
                 ):
        """
        Parameters
        ----------
        volume: {torch.Tensor, 'path'} |\
            Tensor of entire volume or path to tensor.
        volume_tensor: torch.Tensor |\
            Tensor of entire volume.
        patch_size: int |\
            Size of patch with which to partition volume into.
        step_size: int |\
            Size of step between adjacent patch origin.
        """
        self.volume=volume
        #self.volume_tensor, self.nifti = self.volprep()
        self.patch_size=patch_size
        self.step_size=step_size
        self.volume_tensor = None
        self.volume_nifti = None
        self.complete_patch_coords = None
        #if pad_ == True:
        #    print('Padding...')
        #    self.volume_tensor= self.pad_volume()
        #if patch_coords_ == True:
        #    print('Computing complete patch coords...')
        #    self.complete_patch_coords = self.patch_coords()
        #self.volume_shape = self.volume_tensor.shape

    def volprep(self,
                binary:bool=False,
                dtype:torch.dtype=torch.float32,
                normalize:bool=True,
                pmin:float=0,
                pmax:float=1,
                vmin:float=0.05,
                vmax:float=0.95,
                device:str="cuda"
                ) -> tuple:
        """
        Prepare volume.

        Parameters
        ----------
        path : str
            Path to (nifti)
        binary : bool
            Binarize volume (values >= 1 set to 1, values < 1 set to 0)
        dtype : torch.dtype
            Load volume as this data type
        pmin : float
            Lower end quantile for histogram adjustment
        pmax : float
            Higher end quantile for histogram adjustment
        vmin : float
            Shift pmin to this quantile
        vmax : float
            Shift pmax to this quantile
        clip : bool
            Clip histogram that falls outside vmin and vmax
        device : device 
        """
        if isinstance(self.volume, str):
            nifti=nib.load(self.volume)
            tensor=torch.tensor(nifti.get_fdata())
        elif isinstance(self.volume, torch.Tensor):
            tensor = self.volume
            nifti = None
        if binary == True:
            tensor[tensor >= 1] = 1
            tensor[tensor < 1] = 0
        elif binary == False:
            if normalize == True:
                tensor = tensor.unsqueeze(0)
                tensor = cc.QuantileTransform(pmin=pmin, pmax=pmax, vmin=vmin, vmax=vmax, clip=False)(tensor)
                tensor = tensor[0]
        tensor = tensor.to(device).to(dtype)
        self.volume_tensor = tensor
        self.volume_nifti = nifti


    def reshape(self, volume_tensor:torch.Tensor, shape:int=4) -> torch.Tensor:
        """
        Ensure tensor has proper shape

        Parameters
        ----------
        shape: int |\
            Shape that tensor needs to be.
        """
        if len(volume_tensor.shape) < shape:
            volume_tensor = volume_tensor.unsqueeze(0)
            return volume_tensor
        elif len(volume_tensor.shape) == shape:
            return volume_tensor
        else:
            print("Check the shape of your volume tensor plz.")
            exit(0)


    def pad_volume(self, padding_method='replicate'):
        """
        Pad all dimensions of 3D volume according to patch size.

        Parameters
        ----------
        pading_method: {'replicate', 'reflect', 'constant'} |\
            How to pad volume.
        """
        self.padding_method = padding_method
        with torch.no_grad():
            volume_tensor = self.volume_tensor.clone().detach()
            volume_tensor = self.reshape(volume_tensor, 4)
            padding = torch.ones(1, 6, dtype=torch.int) * self.patch_size
            padding = tuple(*padding)
            volume_tensor = torch.nn.functional.pad(
                input=volume_tensor,
                pad=padding,
                mode=padding_method
            ).squeeze()
            self.volume_tensor = volume_tensor


    def patch_coords(self):
        """
        Compute coords for all patches.
        """
        coords = []
        complete_patch_coords = []
        vol_shape = self.volume_tensor.shape
        for dim in range(len(vol_shape)):
            frame_start = np.arange(
                0, vol_shape[dim] - self.patch_size, self.step_size
                )[1:]
            frame_end = [d + self.patch_size for d in frame_start]
            coords.append(list(zip(frame_start, frame_end)))
        if len(coords) == 3:
            for x in coords[0]:
                for y in coords[1]:
                    for z in coords[2]:
                        complete_patch_coords.append([x, y, z])
        elif len(coords) == 2:
            for x in coords[0]:
                for y in coords[1]:
                    complete_patch_coords.append([x, y])  
        self.complete_patch_coords = complete_patch_coords


class OctVolume(Dataset):
    """
    Load OCT volume and partition into patches.
    """
    def __init__(self, path:str, patch_size:int=256, step_size:int=256):
        """
        Parameters
        ---------
        path : str
            Path to OCT volume
        patch_size : int
            Size of patches that OCT volume will be partitioned into
        step_size : int
            Stride length between patches
        """
        self.patch_size = patch_size
        self.step_size = step_size
        vol = VolumeUtils(path, patch_size=patch_size, step_size=step_size)
        vol.volprep(normalize=False)
        vol.pad_volume()
        vol.patch_coords()
        self.volume_tensor = vol.volume_tensor
        self.volume_nifti = vol.volume_nifti
        self.complete_patch_coords = vol.complete_patch_coords
        self.imprint_tensor = torch.zeros(
            self.volume_tensor.shape, dtype=torch.float32, device='cuda'
            )

    def __len__(self):
        return len(self.complete_patch_coords)
    
    def __getitem__(self, idx:int, prediction:bool=False, trainee=None,
                    device:str='cuda') -> tuple:
        """
        Load patch and predict on it.

        Parameters
        ----------
        idx : int
            Patch ID to load and predict on
        prediction : bool
            Choose whether or not to predict on patch
        trainee : trainee
            Trainee used to make the prediction
        device : {'cpu', 'cuda'}
            Device to make predictions
        """
        # Load patch coords
        working_patch_coords = self.complete_patch_coords[idx]
        # Generating slices for easy handling
        x_slice = slice(*working_patch_coords[0])
        y_slice = slice(*working_patch_coords[1])
        z_slice = slice(*working_patch_coords[2])
        # Loading patch via coords and detaching from tracking
        patch = self.volume_tensor[x_slice, y_slice, z_slice].detach().to(device)
        if prediction == True:
            # Make the prediction
            prediction = trainee(patch.unsqueeze(0).unsqueeze(0))
            prediction = torch.sigmoid(prediction).squeeze().squeeze().detach()
            # Add prediction to whole-volume imprint tensor
            self.imprint_tensor[x_slice, y_slice, z_slice] += prediction
            return patch, prediction
        elif prediction == False:
            return patch


    def predict(self, trainee):
        """
        Predict on all patches.

        Parameters
        ----------
        trainee : trainee
        """
        length = self.__len__()
        print("Predicting on", length, 'patches')
        with torch.no_grad():
            # Loop through all patch coordinates
            for i in range(length):
                # Predict on patch
                self.__getitem__(i, prediction=True, trainee=trainee)
                # Log to console
                sys.stdout.write(f"\rPrediction {i + 1}/{length}")
                sys.stdout.flush()
            
            # Removing padding on volume and imprint tensors
            s = slice(self.patch_size, -self.patch_size)
            self.volume_tensor = self.volume_tensor[s, s, s]
            self.imprint_tensor = self.imprint_tensor[s, s, s]
            
            # Averaging prediction based on step size
            factors = {256 : 0, 128: 1, 64: 2, 32: 3, 16:4}
            averaging_factor = 1 / (8 ** factors[self.step_size])
            print("\nAveraging by:", averaging_factor)
            self.imprint_tensor = self.imprint_tensor * averaging_factor


def optionsStuff(options:dict, paths:dict):
    if isinstance(paths["ground_truth"], str):
        pass


class Thresholding(object):
    """
    Decide if and how to threshold. Perform thresholding.
    """
    def __init__(self, prediction_tensor:torch.Tensor,
                 ground_truth_tensor:torch.Tensor,
                 threshold:{float, False, 'auto'}=0.5,
                 compute_accuracy:bool=False
                 ):
        """
        Parameters
        ----------
        prediction_tensor : tensor[float]
            Tensor of prediction volume.
        ground_truth_tensor : tensor[bool]
            Ground truth tensor.
        threshold : {float, False, 'auto'}
            Intensity value at which to threshold. If False, return prob map.
        compute_accuracy : bool
            If true, compute accuracy and print to console.
        """
        self.prediction_tensor = prediction_tensor
        self.ground_truth_tensor = ground_truth_tensor
        self.threshold = threshold
        self.compute_accuracy = compute_accuracy

    def apply(self):
        """
        Run thresholding.
        """
        if self.threshold == False:
            # Return the unaltered probability map
            print("\nNot thresholding...")
            return self.prediction_tensor, None, None
        elif isinstance(self.threshold, float):
            print('\nApplying fixed threshold...')
            return self.fixedThreshold()
        elif self.threshold == 'auto' and isinstance(self.ground_truth_tensor,
                                                   torch.Tensor):
            return self.autoThreshold()
        else:
            print("\nCan't do the thresholding. Check your settings")
            exit(0)
        

    def autoThreshold(self, start:float=0.05, stop:float=0.95, step:float=0.05):
        """
        Auto threshold volume.

        Parameters
        ----------
        start : float
            Intensity to begin thresholding
        stop : float
            Intensity to stop thresholding
        step : float
            Increase from start to stop with this step size
        """
        threshold_lst = np.arange(start, stop, step)
        accuracy_lst = []

        for thresh in threshold_lst:
            temp = self.prediction_tensor.clone()
            temp[temp >= thresh] = 1
            temp[temp <= thresh] = 0
            accuracy = dice(temp, self.ground_truth_tensor, multiclass=False)
            accuracy_lst.append(accuracy)

        max_accuracy_index = accuracy_lst.index(max(accuracy_lst))
        threshold, accuracy = threshold_lst[max_accuracy_index], accuracy_lst[max_accuracy_index]
        # Now do the actual thresholding
        self.prediction_tensor[self.prediction_tensor >= threshold] = 1
        self.prediction_tensor[self.prediction_tensor <= threshold] = 0
        threshold = round(threshold.item(), 3)
        accuracy = round(accuracy.item(), 3)
        return self.prediction_tensor, threshold, accuracy
    

    def fixedThreshold(self):
        """
        Apply a fixed threshold to intensity volume.
        """
        # Do a fixed threshold
        print("\nApplying a fixed threshold...")
        self.prediction_tensor[self.prediction_tensor >= self.prediction_tensor] = 1
        self.prediction_tensor[self.prediction_tensor <= self.prediction_tensor] = 0
        if self.compute_accuracy == True:
            accuracy = dice(self.prediction_tensor, self.ground_truth_tensor, multiclass=False)
            accuracy = round(accuracy.item(), 3)
        elif self.compute_accuracy == False:
            accuracy = None
        else:
            print("Look, do you want me to compute the accuracy or not!")
            exit(0)
        return self.prediction_tensor, self.threshold, accuracy

#Thresholding()

def thresholding(
    prediction_tensor:torch.Tensor,
    ground_truth_tensor=None,
    threshold:bool=True,
    auto_threshold:bool=True,
    fixed_threshold:float=0.5,
    compute_accuracy:bool=True
    ) -> tuple:
    
    auto_threshold_settings = {
        "start": 0.05,
        "stop": 0.95,
        "step": 0.05,  
    }

    #out_filename = f"prediction_stepsz{step_size}"

    if threshold == True:
        if auto_threshold == True:
            # Decide if we can even do threshold
            if ground_truth_tensor is None:
                # Can't threshold because there was no gt tensor
                print("\nCan't threshold! You didn't give me a ground truth tensor!")
            elif isinstance(ground_truth_tensor, torch.Tensor):
                # All good. Go on to auto thresholding
                print("\nAuto thresholding...")
                threshold_lst = np.arange(
                    auto_threshold_settings["start"],
                    auto_threshold_settings['stop'],
                    auto_threshold_settings['step']
                    )
                accuracy_lst = []
                for thresh in threshold_lst:
                    temp = prediction_tensor.clone()
                    temp[temp >= thresh] = 1
                    temp[temp <= thresh] = 0
                    accuracy = dice(temp, ground_truth_tensor, multiclass=False)
                    accuracy_lst.append(accuracy)
                max_index = accuracy_lst.index(max(accuracy_lst))
                threshold, accuracy = threshold_lst[max_index], accuracy_lst[max_index]
                # Now do the actual thresholding
                prediction_tensor[prediction_tensor >= threshold] = 1
                prediction_tensor[prediction_tensor <= threshold] = 0

                threshold = round(threshold.item(), 3)
                accuracy = round(accuracy.item(), 3)
                return prediction_tensor, threshold, accuracy
            
        elif auto_threshold == False:
            # Do a fixed threshold
            print("\nApplying a fixed threshold...")
            prediction_tensor[prediction_tensor >= fixed_threshold] = 1
            prediction_tensor[prediction_tensor <= fixed_threshold] = 0
            if compute_accuracy == True:
                accuracy = dice(prediction_tensor, ground_truth_tensor, multiclass=False)
                accuracy = round(accuracy.item(), 3)
            else:
                accuracy = None
            return prediction_tensor, fixed_threshold, accuracy
    elif threshold == False:
        # Return a prob map
        print("\nNot thresholding...")
        return prediction_tensor, None, None


def volume_stats(tensor, n:int=150, stats:bool=True, zero=False, unique=False, a:int=None, b:int=None, step:float=None):    
    if stats:
        print("\nShape:", tensor.shape)
        print("dtype:", tensor.dtype)
        print("\nVolume Statistics:",'#' * 20)
        print("Mean:", round(tensor.mean().item(), 3))
        print("Median:", round(tensor.median().item(), 3))
        print("StDev:", round(tensor.std().item(), 3))
        print(f"Range: [{round(tensor.min().item(), 3)}, {round(tensor.max().item(), 3)}]")
        # Quantiles
        print("2nd Percentile:", round(torch.quantile(tensor, 0.02).item(), 3))
        print("25th Percentile:", round(torch.quantile(tensor, 0.25).item(), 3))
        print("75th Percentile:", round(torch.quantile(tensor, 0.75).item(), 3))
        print("98th Percentile:", round(torch.quantile(tensor, 0.98).item(), 3))

    img = tensor.to('cpu').numpy().squeeze()
    if a is None:
        a = pymath.floor(img.min())
    if b is None:
        b = pymath.ceil(img.max()) + 2
    if step is None:
        step = 1

    if unique:
        print(np.unique(img))
    else:
        pass

    # Histogram
    #frequency, intensity = np.histogram(img, bins=np.arange(a, b, step))
    # Figure
    #plt.figure()
    #f, axarr = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    #axarr = axarr.flatten()
    #axarr[0].imshow(img[n], cmap='gray')
    #axarr[1].bar(intensity[:-1], frequency, width=0.1)


class PathTools(object):
    """
    Class to handle paths.
    """
    def __init__(self, path:str):
        """
        Parameters
        ----------
        path : str
            Path to deal with. 
        """
        self.path = path


    def destroy(self):
        """
        Delete all files and subdirectories.
        """
        shutil.rmtree(path=self.path)


    def makeDir(self):
        """
        Make new directory. Delete then make again if dir exists.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            self.destroy()
            os.makedirs(self.path)
    

    def patternRemove(self, pattern):
        """
        Remove file in self.path that contains pattern

        Parameters
        ----------
        pattern : str
            Pattern to match to. Examples: {*.nii, *out*, 0001*}
        """
        regex = [
            f"{self.path}/**/{pattern}",
            f"{self.path}/{pattern}"
        ]
        for expression in regex:
            try:
                [os.remove(hit) for hit in glob.glob(
                    expression, recursive=True
                    )]
            except:
                pass

class JsonTools(object):
    """
    Class for handling json files.
    """
    def __init__(self):
        """
        Parameters
        ----------
        path : str
            Path to json file.
        """
    
    def log(self, dict, path):
        """
        Save Python dictionary as json file.

        Parameters
        ----------
        dict : dict
            Python dictionary to save as json.
        path : str
            Path to new json file to create.
        """
        self.json_object = json.dumps(dict, indent=4)
        file = open(path, 'x+')
        file.write(self.json_object)
        file.close()