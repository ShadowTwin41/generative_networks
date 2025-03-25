import os
import torch
import numpy as np
from monai.data import CSVDataset, CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd,
    CopyItemsd,
    CropForegroundd,
    SpatialCropd,
    ToTensord,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandSpatialCropd,
    Lambda
)
from utils.crop_scan_center_in_tumour import CropScanCenterInTumour


def set_contrast_tensor(d):
    if 'contrast' in d: 
        if d['contrast']==0:
            no_contrast_tensor = np.ones_like(d['label'])
            contrast_tensor = np.zeros_like(d['label'])
        elif d['contrast']==1:
            no_contrast_tensor = np.zeros_like(d['label'])
            contrast_tensor = np.ones_like(d['label'])
        else:
            raise ValueError(f"Wrong contrast value: {d['contrast']}")
        d["no_contrast_tensor"] = no_contrast_tensor
        d["contrast_tensor"] = contrast_tensor
        d["scan_volume_crop_pad"] = d["scan_ct"] # Changing the name to work like when no data augmentation is used.
        d["label_crop_pad"] = d["label"] # Changing the name to work like when no data augmentation is used.
    return d

def get_loader(args): 
    NUM_WORKERS = int(args.num_workers)
    CSV_PATH = args.csv_path
    print(f"CSV_PATH: {CSV_PATH}")

    if ("brats" in args.dataset.lower()) or ("hnn" in args.dataset.lower()):
        if ("brats" in args.dataset.lower()) and ("2023" in args.dataset.lower()):
            print(f"Using dataset: BRATS_2023")
            from utils.convert_to_multi_channel_based_on_brats_classes import ConvertToMultiChannelBasedOnBratsGliomaClasses2023d as LABEL_TRANSFORM
        elif ("hnn" in args.dataset.lower()):
            from utils.convert_head_n_neck_cancer import ConvertHeadNNeckCancerd as LABEL_TRANSFORM

        args.modality = args.modality.lower()
        if args.modality == "t1ce":
            scan_name = "scan_t1ce"
            col_names = ['scan_t1ce', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
            col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}
        elif args.modality == "t1":
            scan_name = "scan_t1"
            col_names = ['scan_t1', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
            col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}
        elif args.modality == "t2":
            scan_name = "scan_t2"
            col_names = ['scan_t2', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
            col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}
        elif args.modality == "flair":
            scan_name = "scan_flair"
            col_names = ['scan_flair', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']
            col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}}  
        elif args.modality == "ct":
            scan_name = "scan_ct"
            col_names = ['scan_ct', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size', 'contrast']
            col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}, 'contrast': {'type': int}}      
        print(f"Scan Modality: {scan_name}")

    else:
        raise ValueError("The dataset must be from BraTS: BRATS_2023 or HNN")

    train_transforms = [
                    LoadImaged(keys=[scan_name, 'label'], image_only=False),
                    EnsureChannelFirstd(keys=[scan_name, "label"]),
                    EnsureTyped(keys=[scan_name, "label"]),
                    CopyItemsd(keys=[scan_name], names=[f"{scan_name}_origin"]),
                    ScaleIntensityRanged(keys=[f"{scan_name}_origin"], a_min=int(args.clip_min), a_max=int(args.clip_max), b_min=int(args.clip_min), b_max=int(args.clip_max), clip=True),
                    LABEL_TRANSFORM(keys="label"),
                ] 
    if args.use_dilation:
        new_keys = ['scan_volume_crop_pad', 'label_crop_pad_dilated', 'label_crop_pad']
        interpolation_mode=['trilinear', 'nearest', 'nearest']
    else:
        new_keys = ['scan_volume_crop_pad', 'label_crop_pad']
        interpolation_mode=['trilinear', 'nearest']

    if args.use_data_augmentation:
        #train_transforms.append(
        #    CropScanCenterInTumour(keys=scan_name, dilation=args.use_dilation, translate_range=28)
        #    )  
        #  
        col_names = ['scan_ct', 'label', 'contrast']
        col_types = None
        train_transforms.append(
            RandSpatialCropd(keys=[scan_name,'label'], roi_size=[128,128,128])
            )
        train_transforms.append(
            Lambda(set_contrast_tensor)
            )
        train_transforms.append(
            RandFlipd(keys=new_keys, spatial_axis=0, prob=0.1, lazy=True)
            )
        train_transforms.append(
            RandFlipd(keys=new_keys, spatial_axis=1, prob=0.1, lazy=True)
            )
        train_transforms.append(
            RandFlipd(keys=new_keys, spatial_axis=2, prob=0.1, lazy=True)
            )
        # Rotate 90 degrees
        train_transforms.append(
                        RandRotate90d(keys=new_keys, prob=0.1, max_k=3, lazy=True)
        )

        # Based on file:///Users/andreferreira/Downloads/s10462-023-10453-z.pdf and https://arxiv.org/pdf/2006.06676.pdf
        # rotate 45 degrees
        # scale_range (-0.1, 0.1) -> zoom!
        # shear_range (-0.1, 0.1)
        train_transforms.append(
            RandAffined(
                        keys=new_keys,
                        prob=0.1,
                        rotate_range=((-np.pi/4,np.pi/4),(-np.pi/4,np.pi/4),(-np.pi/4,np.pi/4)), # 6 degrees
                        #translate_range=(16,16,16), 
                        scale_range=((-0.2,0.2),(-0.2,0.2),(-0.2,0.2)),
                        shear_range=((-0.2,0.2),(-0.2,0.2),(-0.2,0.2)),
                        padding_mode="border",
                        mode=interpolation_mode,
                        lazy=True,
                        )
        )
        train_transforms.append(
            ScaleIntensityRanged(keys=[scan_name, "scan_volume_crop_pad"], a_min=int(args.clip_min), a_max=int(args.clip_max), b_min=-1.0, b_max=1.0, clip=True)
        )
        train_transforms.append(
            ToTensord(keys=[scan_name, 'no_contrast_tensor', 'contrast_tensor', 'scan_volume_crop_pad', 'label', 'label_crop_pad'])
            )
    else:
        train_transforms.append(
            CropScanCenterInTumour(keys=scan_name, dilation=args.use_dilation, translate_range=None)
            )       
        train_transforms.append(
            ScaleIntensityRanged(keys=[scan_name], a_min=int(args.clip_min), a_max=int(args.clip_max), b_min=-1.0, b_max=1.0, clip=True)
        )
        if args.use_dilation:
            train_transforms.append(ToTensord(keys=[scan_name, 'no_contrast_tensor', 'contrast_tensor', 'label_crop_pad_dilated', 'scan_volume_crop', 'scan_volume_crop_pad', 'label', 'label_crop_pad', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']))
        else:
            train_transforms.append(ToTensord(keys=[scan_name, 'no_contrast_tensor', 'contrast_tensor', 'scan_volume_crop', 'scan_volume_crop_pad', 'label', 'label_crop_pad', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']))
        
    final_train_transforms = Compose(train_transforms)
    
    # USING THE WHOLE DATASET
    if args.use_data_augmentation:
        print(f"Using data augmentation on the fly!")
        train_CSVdataset = CSVDataset(src=CSV_PATH, col_names=col_names, col_types=col_types, transform=final_train_transforms)
        train_loader = DataLoader(train_CSVdataset, batch_size=int(args.batch_size), num_workers=NUM_WORKERS, drop_last=True, shuffle=True, collate_fn=pad_list_data_collate) 
    else:
        if args.mode=="c_sample":
            train_CSVdataset = CSVDataset(src=CSV_PATH, col_names=col_names, col_types=col_types)
            train_CSVdataset = CacheDataset(train_CSVdataset, transform=final_train_transforms, cache_rate=0, num_workers=8, progress=True)  
            train_loader = DataLoader(train_CSVdataset, batch_size=int(args.batch_size), num_workers=NUM_WORKERS, drop_last=True, shuffle=False, collate_fn=pad_list_data_collate)
        else:
            print(f"Not using data augmentation! Loading all data to memory")
            train_CSVdataset = CSVDataset(src=CSV_PATH, col_names=col_names, col_types=col_types) 
            train_CSVdataset = CacheDataset(train_CSVdataset, transform=final_train_transforms, cache_rate=0, num_workers=8, progress=True)  
            train_loader = DataLoader(train_CSVdataset, batch_size=int(args.batch_size), num_workers=8, drop_last=True, shuffle=True, collate_fn=pad_list_data_collate)
    print(f"Number of training images: {len(train_CSVdataset)}")
    print(f'Dataset training: number of batches: {len(train_loader)}')
    print("Leaving the data loader. Good luck!") 
    return train_loader