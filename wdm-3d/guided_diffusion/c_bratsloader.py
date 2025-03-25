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
    ScaleIntensityRangePercentilesd,
    )
from utils.data_loader_utils import ConvertToMultiChannelBasedOnBratsClasses2023d, QuantileAndScaleIntensityd

class c_BraTSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, batch_size, num_workers, image_key, normalize=None, mode='train', img_size=256, no_seg=False):
        print(f"directory: {directory}")

        self.directory = directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.mode = mode
        self.img_size = img_size
        self.no_seg=no_seg
        self.image_key=image_key

    def generate_detection_train_transform(self,
        image_key,
        label_key,
        image_size,
        no_seg=False,
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
        if len(image_key.split("_"))>1:
            in_keys = []
            all_image_keys = []
            for modal in image_key.split("_"):
                in_keys.append(modal)
                all_image_keys.append(modal)
            in_keys.append(label_key)
        else:
            in_keys = [image_key, label_key]
            all_image_keys = [image_key]
        
        compute_dtype = torch.float32
        train_transforms = Compose(
                [
                    LoadImaged(keys=in_keys, meta_key_postfix="meta_dict", image_only=False),
                    EnsureChannelFirstd(keys=in_keys),
                    EnsureTyped(keys=in_keys, dtype=torch.float32),
                    Orientationd(keys=in_keys, axcodes="RAS"),
                    ResizeWithPadOrCropd(
                            keys=in_keys,
                            spatial_size=image_size,
                            mode="constant",
                            value=0
                        ),
                    QuantileAndScaleIntensityd(keys=all_image_keys), # a_min=-1, a_max=1),
                    ConvertToMultiChannelBasedOnBratsClasses2023d(
                        keys=[label_key], no_seg=no_seg,
                    ),
                    EnsureTyped(keys=in_keys, dtype=compute_dtype)
                ]
            )
        return train_transforms
    

    def get_loader(self, directory, batch_size, num_workers, normalize=None, mode='train', img_size=256, no_seg=False):
        """
        ARGS:
            directory: root directory for the dataset
            test_flag: Batch size
            
        RETURN:
            train_loader: data loader
            train_data: dict of the data loaded 
        """

        data_split_json = os.path.join('/'.join(directory.split("/")[0:-1]), "BraTS2023_GLI_data_split.json")
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
                image_key = self.image_key,
                label_key = "seg",
                image_size = (img_size,img_size,img_size),
                no_seg=no_seg,
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
        dl, ds = self.get_loader(self.directory, self.batch_size, self.num_workers, self.normalize, self.mode, self.img_size, self.no_seg)
        return dl, ds