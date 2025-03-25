import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
import json
import sys
#sys.path.insert(1, "/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d")

from monai.data import load_decathlon_datalist, DataLoader, CacheDataset
from monai.transforms import (
    Compose, 
    LoadImaged,
    EnsureChannelFirstd, 
    EnsureTyped,
    Orientationd,
    Resized,
    ScaleIntensityRanged, 
    ResizeWithPadOrCropd,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandAdjustContrastd,
    RandRotate90d,
    CopyItemsd
    )
from utils.data_loader_utils import ConvertToMultiChannel_BackandForeground_Contrastd

class RandAffined_and_Crop(RandAffined):
  def __call__(self, data, lazy=None):
    # Call the original RandAffine transformation
    output = super().__call__(data, lazy=lazy)
    prob_here = self.R.rand()
    if prob_here>=0.5:
      if data['contrast']==0:
        output['image'] = torch.where(output['seg'][0:1,:,:,:] == 1, output['image_copy'], torch.tensor(-1.0, dtype=output['image_copy'].dtype))
      elif data['contrast']==1:
        output['image'] = torch.where(output['seg'][1:2,:,:,:] == 1, output['image_copy'], torch.tensor(-1.0, dtype=output['image_copy'].dtype))
    elif prob_here<0.5:
      if data['contrast']==0:
        output['image'] = torch.where(output['seg'][0:1,:,:,:] == 1, output['image'], torch.tensor(-1.0, dtype=output['image'].dtype))
      elif data['contrast']==1:
        output['image'] = torch.where(output['seg'][1:2,:,:,:] == 1, output['image'], torch.tensor(-1.0, dtype=output['image'].dtype))
    return output

class HnNVolumes(torch.utils.data.Dataset):
    def __init__(self, args, directory, batch_size, num_workers, normalize=None, mode='train', img_size=256, no_seg=False, full_background=False, clip_min=None, clip_max=None):
        print(f"directory: {directory}")
        self.args = args
        self.directory = directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.mode = mode
        self.img_size = img_size
        self.no_seg=no_seg
        self.full_background=full_background
        self.clip_min=clip_min
        self.clip_max=clip_max

    def generate_detection_train_transform(self,
        image_key,
        label_key,
        image_size,
        clip_min,
        clip_max, 
        no_seg=False,
        full_background=False,
    ):
        """
        Generate training transform for the GAN.

        ARGS:
            image_key: the key to represent images in the input json files
            label_key: the key to represent labels in the input json files
            image_size: final image size for resizing 

        RETURN:
            training transform for the GAN
        """
        print(f"clip_min: {clip_min}")
        print(f"clip_max: {clip_max}")
        print(f"image_key: {image_key}")
        compute_dtype = torch.float32
        
        train_transforms = [
                LoadImaged(keys=[image_key, label_key], meta_key_postfix="meta_dict", image_only=False),
                EnsureChannelFirstd(keys=[image_key, label_key]),
                EnsureTyped(keys=[image_key, label_key], dtype=torch.float32),
                Orientationd(keys=[image_key, label_key], axcodes="RAS"),
                ScaleIntensityRanged(keys=[image_key], a_min=float(clip_min), a_max=float(clip_max), b_min=-1.0, b_max=1.0, clip=True),
                ResizeWithPadOrCropd(
                        keys=[image_key, label_key],
                        spatial_size=image_size,
                        mode="constant",
                        value=-1 # The value was -1 originally
                    ),
                ConvertToMultiChannel_BackandForeground_Contrastd(
                        keys=[label_key], no_seg=no_seg, full_background=full_background
                        )
            ]
        if self.args.ROI_DataAug:
            train_transforms.append(
                CopyItemsd(
                    keys=[image_key], times=1, names=[f"{image_key}_copy"], 
                )
            )
            train_transforms.append(
                RandAffined_and_Crop(keys=[image_key, label_key],
                    prob=1, 
                    scale_range=((-0.05, 0.2), (-0.05, 0.2), (-0.05, 0.2)), 
                    mode="nearest", 
                    padding_mode="zeros")
        )
        train_transforms.append(EnsureTyped(keys=[image_key, label_key], dtype=compute_dtype))
        train_transforms_final =  Compose(train_transforms)
        return train_transforms_final

    def get_loader(self, directory, batch_size, num_workers, normalize=None, mode='train', img_size=256, no_seg=False, full_background=False, clip_min=None, clip_max=None):
        """
        ARGS:
            directory: root directory for the dataset
            test_flag: Batch size
            
        RETURN:
            train_loader: data loader
            train_data: dict of the data loaded 
        """

        data_split_json = os.path.join(directory, "data_split.json")
        if mode == 'train':
            data_list_key = "training"
            shuffle = True
        elif mode == 'test':
            data_list_key = "test"
            shuffle = False
        else:
            raise ValueError(f"Chosen mode not available: {mode}. Available modes are train or test.")


        # Get train transforms
        transforms = self.generate_detection_train_transform(
                image_key = "image",
                label_key = "seg",
                image_size = (img_size,img_size,img_size),
                clip_min=clip_min,
                clip_max=clip_max,
                no_seg=no_seg,
                full_background=full_background,
            )

        # Get training data dict 
        data_set = load_decathlon_datalist(
                data_split_json,
                is_segmentation=True,
                data_list_key="training",
                base_dir=directory,
            )

        print(f"Training cases: {len(data_set)}")

        print(data_set[-1:])
        print(f"TOTAL cases {len(data_set)}")
        # Creating traing dataset
        ds = CacheDataset( 
            data=data_set, 
            transform=transforms,
            cache_rate=0, 
            copy_cache=False,
            progress=True,
            num_workers=num_workers,
        )
        
        # Creating data loader
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            shuffle=shuffle, 
            #collate_fn=no_collation,
        )

        print(f"Batch size: {batch_size}")
        return dl, ds

    def get_dl_ds(self):
        dl, ds = self.get_loader(self.directory, self.batch_size, self.num_workers, self.normalize, self.mode, self.img_size, self.no_seg, self.full_background, clip_min=self.clip_min, clip_max=self.clip_max)
        return dl, ds

