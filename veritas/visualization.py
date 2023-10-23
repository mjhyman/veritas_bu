__all__ = [
    'Visualize',
    'Confusion'
]
# Standard Imports
import torch
import numpy as np
import tifffile
# Custom Imports
from veritas.data import RealOct

class Visualize(object):
    """
    Base class for visualization.
    """
    def __init__(self, in_path, out_path=None, out_name='movie'):
        self.in_path = in_path
        if out_path is None:
            self.out_path = f"{'/'.join(self.in_path.split('/')[:-1])}/{out_name}.tiff"
        else:
            self.out_path = out_path
        self.vol = self.load_base_vol()


    def load_base_vol(self):
        """
        Load base volume and convert to 3 channels.

        Returns
        -------
        self.vol : tensor
            Base volume.
        """
        vol = RealOct(volume=self.in_path, normalize=True, p_bounds=[0, 1],
            v_bounds=[0, 1], device='cpu', pad_=False).volume_tensor.numpy()
        # Clamping
        vol[vol > 1] = 1
        vol[vol < 0] = 0
        # Converting to 8-bit color
        vol = np.uint8(vol* 255)
        vol = np.stack((vol, vol, vol), axis=-1)
        return vol


    def overlay(self, overlay_tensor:torch.Tensor, name:str, rgb=[0, 0, 255], binary_threshold=None):
        """
        Overlay something onto base volume.

        Parameters
        ----------
        overlay_tensor : tensor[bool]
            Tensor to use as overlay. shape = [x,y,z]
        """
        if isinstance(overlay_tensor, str):
            print(f'Loading overlay {name} from path...\n')
            overlay_tensor = RealOct(
                volume=overlay_tensor, binarize=True, device='cpu',
                dtype=torch.uint8
                ).volume_tensor.numpy()
        elif isinstance(overlay_tensor, torch.Tensor):
            print(f'Using tensor {name} as numpy arr...\n')
            overlay_tensor = overlay_tensor.numpy()
        elif isinstance(overlay_tensor, np.ndarray):
            print(f"Using numpy array {name}...")
        else:
            print(f'Having trouble with overlaying {name}')
            exit(0)
        print(f"{name} max = {overlay_tensor.max()}")
        print(f"{name} min = {overlay_tensor.min()}")

        for i in range(3):
            self.vol[..., i][overlay_tensor == 1] = 0
            self.vol[..., i] += (overlay_tensor * rgb[i])

        
    def make_tiff(self):
        self.vol[self.vol > 255] = 255
        from scipy import ndimage
        self.vol = ndimage.zoom(self.vol, [1, 12, 12, 1], order=0)
        print(f'Saving to: {self.out_path}...')
        tifffile.imwrite(self.out_path, self.vol)


vol_path = '/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/64x64x64_sub-I38_ses-OCT_sample-BrocaAreaS01_OCT-volume.nii'
ground_truth = '/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/64_ground-truth.nii'
prediction = '/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/predictions/64x64x64_sub-I38_ses-OCT_sample-BrocaAreaS01_OCT-volume-prediction_stepsz-32.nii'

#vis = Visualize(vol_path, out_name='prediction')
#vis.overlay(ground_truth, name='ground_truth', rgb=[0, 0, 255])
#vis.overlay(prediction, name='prediction', rgb=[0, 255, 255])
#vis.make_tiff()



class Confusion(object):
    def __init__(self, ground_truth, prediction):
        self.ground_truth = RealOct(volume=ground_truth, binarize=True, device='cpu',
                                dtype=torch.uint8).volume_tensor.numpy()
        self.prediction = RealOct(volume=prediction, binarize=True, device='cpu',
                                dtype=torch.uint8).volume_tensor.numpy()
        
        
    def true_positives(self):
        """
        True positives (yellow)
        """
        out_vol = np.zeros(self.ground_truth.shape, dtype=np.uint8)
        out_vol[self.prediction == 1] += 1
        out_vol[self.ground_truth == 1] += 1
        out_vol[out_vol < 2] = 0
        out_vol[out_vol >= 2] = 1
        rgb = [255, 255, 0]
        return out_vol, rgb


    def false_positives(self):
        """
        False positives (red)
        """
        out_vol = np.zeros(self.ground_truth.shape, dtype=np.uint8)
        out_vol[self.prediction == 1] = 1
        out_vol[self.ground_truth == 1] = 0
        #out_vol = self.prediction - self.ground_truth
        #out_vol[out_vol <= 0] = 0
        #out_vol[out_vol >= 1] = 1
        rgb = [255, 0, 0]
        return out_vol, rgb
    

    def false_negatives(self):
        """
        False negatives (green)
        """
        out_vol = np.zeros(self.ground_truth.shape, dtype=np.uint8)
        out_vol[self.ground_truth >= 1] = 1
        out_vol[self.prediction == 1] = 0
        rgb = [0, 255, 0]
        return out_vol, rgb
    

vol_path = '/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/unet-validation-volumes/I48/I48_vol-2.nii'
out_name = vol_path.split('/')[-1].strip('.nii')
out_path = None #"/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/unet-validation-volumes/I38/I38_vol-1.tiff"

# true positive = yellow, false positive = red, false negative = green
vis = Visualize(vol_path, out_name=out_name)
#confusion = Confusion(ground_truth, prediction)

#tp, tp_rgb = confusion.true_positives()
#vis.overlay(tp, name='true_positives', rgb=tp_rgb)

#fp, fp_rgb = confusion.false_positives()
#vis.overlay(fp, name='false_positives', rgb=fp_rgb)

#fn, fn_rgb = confusion.false_negatives()
#vis.overlay(fn, name='false_negatives', rgb=fn_rgb)
vis.make_tiff()