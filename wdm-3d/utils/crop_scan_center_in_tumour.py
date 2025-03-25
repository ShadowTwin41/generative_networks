import math
import torch
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from torch import clone as clone
import numpy as np
import random
import scipy

class CropScanCenterInTumour(MapTransform):
    """
    Crops the labels and the scan based on the label's geometric center.
    """
    def __init__(self, keys: KeysCollection, dilation=False, translate_range=None):
        super().__init__(keys)
        self.keys = keys
        self.dilation = dilation
        self.translate_range = translate_range
    def __call__(self, data):
        d = dict(data)
        scan_volume = d[self.keys]
        _, max_x, max_y, max_z = scan_volume.shape
        scan_volume_crop = clone(scan_volume)
        label = d["label"]
        label_crop = clone(label)

        x_extreme_dif = d["x_extreme_max"] - d["x_extreme_min"]
        y_extreme_dif = d["y_extreme_max"] - d["y_extreme_min"]
        z_extreme_dif = d["z_extreme_max"] - d["z_extreme_min"]

        x_pad = (128 - x_extreme_dif) / 2
        y_pad = (128 - y_extreme_dif) / 2
        z_pad = (128 - z_extreme_dif) / 2

        if x_pad < 0:
            C_x = -0.5
        else:
            C_x = 0.5

        if y_pad < 0:
            C_y = -0.5
        else:
            C_y = 0.5

        if z_pad < 0:
            C_z = -0.5
        else:
            C_z = 0.5

        x_base = d["x_extreme_min"] - int(x_pad)
        x_top = d["x_extreme_max"] + int(x_pad+C_x) 
        y_base = d["y_extreme_min"] - int(y_pad) 
        y_top = d["y_extreme_max"] + int(y_pad+C_y) 
        z_base = d["z_extreme_min"] - int(z_pad) 
        z_top = d["z_extreme_max"] + int(z_pad+C_z)

        if self.translate_range is not None:
            # Randomly shifts the crop.
            # It chooses randomly a value between the value chosen and the negative of the value choosen
            # It chooses randomly for each axis
            here_translate_range = random.randint(-self.translate_range, self.translate_range)
            x_base+=here_translate_range 
            x_top+=here_translate_range 
            here_translate_range = random.randint(-self.translate_range, self.translate_range)
            y_base+=here_translate_range 
            y_top+=here_translate_range 
            here_translate_range = random.randint(-self.translate_range, self.translate_range)
            z_base+=here_translate_range 
            z_top+=here_translate_range 

        
        # Verifying the need for padding
        x_base_pad = 0
        y_base_pad = 0
        z_base_pad = 0
        x_top_pad = 0
        y_top_pad = 0
        z_top_pad = 0

        if x_base < 0:
            x_base_pad = -x_base
            x_base = 0
            
        if y_base < 0:
            y_base_pad = -y_base
            y_base = 0
            
        if z_base < 0:
            z_base_pad = -z_base
            z_base = 0
            
        if x_top > max_x:
            x_top_pad = x_top-max_x
            x_top = max_x
            
        if y_top > max_y:
            y_top_pad = y_top-max_y
            y_top = max_y
            
        if z_top > max_z:
            z_top_pad = z_top-max_z
            z_top = max_z
        ##################################
        # Crop the label
        label_crop = label_crop[:, x_base : x_top, y_base : y_top, z_base : z_top]
        
        # Crop and Normalise the scan
        scan_volume_crop = scan_volume_crop[:, x_base : x_top, y_base : y_top, z_base : z_top]

        if torch.sum(scan_volume_crop)==0:
            raise Exception("It is an empty case")

        scan_volume_crop = self.rescale_array(arr=scan_volume_crop, minv=-1, maxv=1)
        d["scan_volume_crop"] = scan_volume_crop
        
        # Scan and label with padding for 128, 128, 128 (if needed)
        scan_volume_crop_pad = clone(scan_volume_crop)
        scan_volume_crop_pad = np.pad(scan_volume_crop_pad, pad_width=((0,0), (x_base_pad,x_top_pad), (y_base_pad,y_top_pad), (z_base_pad,z_top_pad)), mode='constant', constant_values=(-1, -1))
        label_crop_pad = clone(label_crop)
        label_crop_pad = np.pad(label_crop_pad, pad_width=((0,0), (x_base_pad,x_top_pad), (y_base_pad,y_top_pad), (z_base_pad,z_top_pad)), mode='constant', constant_values=(0, 0))

        scan_volume_crop_pad = self.rescale_array_numpy(arr=scan_volume_crop_pad, minv=-1, maxv=1)

        # Create background contrast or not
        if 'contrast' in d: 
            if d['contrast']==0:
                no_contrast_tensor = np.ones_like(scan_volume_crop_pad)
                contrast_tensor = np.zeros_like(scan_volume_crop_pad)
            elif d['contrast']==1:
                no_contrast_tensor = np.zeros_like(scan_volume_crop_pad)
                contrast_tensor = np.ones_like(scan_volume_crop_pad)
            else:
                raise ValueError(f"Wrong contrast value: {d['contrast']}")
        
        d[self.keys] = scan_volume
        d["no_contrast_tensor"] = no_contrast_tensor
        d["contrast_tensor"] = contrast_tensor
        d["scan_volume_crop_pad"] = scan_volume_crop_pad
        d["label_crop"] = label_crop  
        d["label_crop_pad"] = label_crop_pad

        if self.dilation:
            label_crop_pad_dilated = label_crop_pad[0]
            label_crop_pad_dilated = scipy.ndimage.binary_dilation(input=label_crop_pad_dilated, structure=scipy.ndimage.generate_binary_structure(3, 2), iterations=5)
            label_crop_pad_dilated = np.expand_dims(label_crop_pad_dilated, axis=0)
            d["label_crop_pad_dilated"] = label_crop_pad_dilated
        
        return d

    def rescale_array(self, arr, minv, maxv): #monai function adapted
        """
        Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
        """
        mina = torch.min(arr)
        maxa = torch.max(arr)
        if mina == maxa:
            return arr * minv
        # normalize the array first
        norm = (arr - mina) / (maxa - mina) 
        # rescale by minv and maxv, which is the normalized array by default 
        return (norm * (maxv - minv)) + minv  

    def rescale_array_numpy(self, arr, minv, maxv): #monai function adapted
        """
        Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
        """
        mina = np.min(arr)
        maxa = np.max(arr)
        if mina == maxa:
            return arr * minv
        # normalize the array first
        norm = (arr - mina) / (maxa - mina) 
        # rescale by minv and maxv, which is the normalized array by default 
        return (norm * (maxv - minv)) + minv  
    
    