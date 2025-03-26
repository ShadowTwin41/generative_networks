"""
A script for sampling from a diffusion model for unconditional image generation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import warnings
import random
import torch.nn.functional as F
import torch as th
from monai.transforms import Resize
from monai.transforms import Compose, LoadImage, CropForeground, EnsureChannelFirst, ResizeWithPadOrCrop, ScaleIntensityRange
# Local imports
import sys
sys.path.append(".")
from utils import data_inpaint_utils
from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.c_script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict,
                                          add_dict_to_argparser)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes
from guided_diffusion.hnnloader import HnNVolumes
from guided_diffusion.c_bratsloader import c_BraTSVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from scipy.ndimage import binary_dilation 
from scipy.ndimage import center_of_mass  
import json

def blur_mask_3d(mask, label_crop_pad, blur_factor, blur_type):
    """
    Apply Gaussian blur to a 3D mask.
    
    Args:
        mask (torch.Tensor): The mask tensor of shape (1, 1, H, W, D).
        blur_factor (int): Kernel size for the Gaussian blur. Should be odd.

    Returns:
        torch.Tensor: Blurred mask.
    """
    # Ensure blur_factor is odd
    if blur_factor % 2 == 0:
        blur_factor += 1
    
    # Create a 3D Gaussian kernel
    sigma = blur_factor / 12.0  # Rule of thumb for Gaussian kernel
    x = th.linspace(-3, 3, blur_factor)
    kernel_1d = th.exp(-0.5 * x**2 / sigma**2)
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize
    
    # Create 3D kernel from 1D kernels
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel_3d = kernel_3d.to(mask.device)
    kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    # Pad the mask for convolution
    padding = blur_factor // 2
    mask_padded = F.pad(mask, (padding, padding, padding, padding, padding, padding), mode="replicate")
    
    # Apply convolution
    blurred_mask = F.conv3d(mask_padded, kernel_3d, padding=0)
    if blur_type=="edge_blur":
        blurred_mask[label_crop_pad==1] = label_crop_pad[label_crop_pad==1]
    elif blur_type=="full_blur":
        blurred_mask=blurred_mask
    else:
        raise ValueError(f"blur_type must be edge_blur or full_blur not {blurred_mask}")
    return blurred_mask

def rescale_array(arr, minv, maxv): #monai function adapted
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    """
    if isinstance(arr, np.ndarray):
        mina = np.min(arr)
        maxa = np.max(arr)
    elif isinstance(arr, th.Tensor):
        mina = th.min(arr)
        maxa = th.max(arr)
    if mina == maxa:
        return arr * minv
    # normalize the array first
    norm = (arr - mina) / (maxa - mina) 
    # rescale by minv and maxv, which is the normalized array by default 
    return (norm * (maxv - minv)) + minv  

def get_crop_tensors_no_random(healthy_ct_scan_full_res, segmentation, device):
    """
    Selects a random center and crops the volume with that center and shape 128x128x128.
    Arguments:
        healthy_ct_scan_full_res (numpy array): Healthy volume.
        region_to_place_tumour_mask (numpy array): Mask of the region to where the tumour can be placed.
        segmentation (numpy array): Tumour segmentation.
    """
    # Padding the volume so no region ouside of the volume is selected
    healthy_ct_scan_full_res = np.clip(healthy_ct_scan_full_res, -200, 200) # TODO in case of using -1000 and 1000
    healthy_ct_scan_full_res = np.pad(healthy_ct_scan_full_res, pad_width=64, mode='constant', constant_values=-200) # -200 background
    segmentation = np.pad(segmentation, pad_width=64, mode='constant', constant_values=0) # 0 background
   
    # Select center
    centroid = center_of_mass(segmentation)
    random_x, random_y, random_z = int(centroid[0]), int(centroid[1]), int(centroid[2])
    random_voxel_indices = [[int(centroid[0]), int(centroid[1]), int(centroid[2])]] # Not random 
    
    # Crop the full resolution scan and mask
    healthy_ct_scan = healthy_ct_scan_full_res[
        random_x-64:random_x+64,
        random_y-64:random_y+64,
        random_z-64:random_z+64
        ]

    label_crop_pad = segmentation[
        random_x-64:random_x+64,
        random_y-64:random_y+64,
        random_z-64:random_z+64
        ]

    # Keep the original intensities of the cropped region
    healthy_ct_scan_origin_intensities = np.copy(healthy_ct_scan)
    
    # Convert to torch and add two dimentions
    healthy_ct_scan = th.from_numpy(healthy_ct_scan)
    healthy_ct_scan = rescale_array(healthy_ct_scan, minv=-1, maxv=1)
    healthy_ct_scan = healthy_ct_scan.unsqueeze(0).unsqueeze(0).to(device)
    
    healthy_ct_scan_origin_intensities = th.from_numpy(healthy_ct_scan_origin_intensities)
    healthy_ct_scan_origin_intensities = healthy_ct_scan_origin_intensities.unsqueeze(0).unsqueeze(0).to(device)
    label_crop_pad = th.from_numpy(label_crop_pad)
    label_crop_pad = label_crop_pad.unsqueeze(0).unsqueeze(0).to(device)
    healthy_ct_scan_full_res = th.from_numpy(healthy_ct_scan_full_res)
    healthy_ct_scan_full_res = healthy_ct_scan_full_res.unsqueeze(0).unsqueeze(0).to(device)
    
    return healthy_ct_scan_full_res, healthy_ct_scan, healthy_ct_scan_origin_intensities, label_crop_pad, random_voxel_indices

def get_label_condition_no_random(args, case_path, contrast_value, seg_path):
    if args.dataset == 'hnn_tumour_inpainting':
        healthy_ct_scan_full_res = get_tensor(
            args=args,
            file_path=case_path,
            norm=False,
            clip=True
            )
        
        segmentation = get_tensor(
            args=args,
            file_path=seg_path,
            norm=False,
            clip=False
            )
            
        # Get the volumes in Torch tensor and the segmentation
        healthy_ct_scan_full_res, healthy_ct_scan, healthy_ct_scan_origin_intensities, label_crop_pad, random_voxel_indices = get_crop_tensors_no_random(healthy_ct_scan_full_res, segmentation, device="cuda:0")

        if contrast_value==0:
            print(f"without_contrast")
            no_contrast_tensor = th.ones_like(label_crop_pad).cuda()
            contrast_tensor = th.zeros_like(label_crop_pad).cuda()
        elif contrast_value==1:
            print(f"with_contrast")
            contrast_tensor = th.ones_like(label_crop_pad).cuda()
            no_contrast_tensor = th.zeros_like(label_crop_pad).cuda()
        
        # Get random center coordenates
        random_x, random_y, random_z = random_voxel_indices[0][0], random_voxel_indices[0][1], random_voxel_indices[0][2]
        sagittal = (random_x-64, random_x+64)
        coronal = (random_y-64, random_y+64)
        axial = (random_z-64, random_z+64)
        
        return sagittal, coronal, axial, no_contrast_tensor, contrast_tensor, label_crop_pad, healthy_ct_scan, healthy_ct_scan_origin_intensities, healthy_ct_scan_full_res
    else:
        raise Exception(f"Not implemented yet! args.dataset needs to be 'hnn_tumour_inpainting'")

def get_label_condition(args, case_path, contrast_value, seg_path, mask_file_path):
    if args.dataset == 'hnn_tumour_inpainting':
        healthy_ct_scan_full_res = get_tensor(
            args=args,
            file_path=case_path,
            norm=False,
            clip=True
            )
        
        region_to_place_tumour_mask = get_tensor(
            args=args,
            file_path=mask_file_path,
            norm=False,
            clip=False,
            )
        segmentation = get_segmentation(
            file_path=seg_path
            )
            
        # Get the volumes in Torch tensor and the segmentation
        healthy_ct_scan_full_res, healthy_ct_scan, healthy_ct_scan_origin_intensities, label_crop_pad, random_voxel_indices = get_crop_tensors(healthy_ct_scan_full_res, region_to_place_tumour_mask, segmentation, device="cuda:0")

        if contrast_value==0:
            print(f"without_contrast")
            no_contrast_tensor = th.ones_like(label_crop_pad).cuda()
            contrast_tensor = th.zeros_like(label_crop_pad).cuda()
        elif contrast_value==1:
            print(f"with_contrast")
            contrast_tensor = th.ones_like(label_crop_pad).cuda()
            no_contrast_tensor = th.zeros_like(label_crop_pad).cuda()
        
        # Get random center coordenates
        random_x, random_y, random_z = random_voxel_indices[0][0], random_voxel_indices[0][1], random_voxel_indices[0][2]
        sagittal = (random_x-64, random_x+64)
        coronal = (random_y-64, random_y+64)
        axial = (random_z-64, random_z+64)
        
        return sagittal, coronal, axial,no_contrast_tensor, contrast_tensor, label_crop_pad, healthy_ct_scan, healthy_ct_scan_origin_intensities, healthy_ct_scan_full_res
    else:
        raise Exception(f"Not implemented yet! args.dataset needs to be 'hnn_tumour_inpainting'")

def get_tensor(args, file_path, norm, clip):
    """
    Loads the nii.gz file, and normalises if necessary.
    Arguments:
        file_path (str): Path to the nii.gz file.
        norm (str): True for clipping and normalisation.
    Return:
        Numpy array of nii.gz file.
    """
    transforms = [
        LoadImage(image_only=True),
        EnsureChannelFirst()
        ]
    if clip:
        transforms.append(
        ScaleIntensityRange(a_min=int(args.clip_min), a_max=int(args.clip_max), b_min=int(args.clip_min), b_max=int(args.clip_max), clip=True)
        )
    if norm:
        transforms.append(
        ScaleIntensityRange(a_min=int(args.clip_min), a_max=int(args.clip_max), b_min=-1, b_max=1, clip=True)
        )
    apply_transforms = Compose(transforms)
    np_tensor = apply_transforms(file_path)[0].numpy()
    return np_tensor

def get_segmentation(file_path):
    """
    Load the segmentation, crops the foreground and reshape to 128x128x128 using padding.
    This ensures that the segmentation is in the middle of the volume
    Arguments:
        file_path (str): Path to the segmentation file.
    Return:
        Numpy array of the segmentation.
    """
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        CropForeground(select_fn=lambda x: x > 0, margin=0),
        ResizeWithPadOrCrop(spatial_size=(128,128,128))
    ])
    segmentation = transforms(file_path)[0].numpy()
    return segmentation

def get_crop_tensors(healthy_ct_scan_full_res, region_to_place_tumour_mask, segmentation, device):
    """
    Selects a random center and crops the volume with that center and shape 128x128x128.
    Arguments:
        healthy_ct_scan_full_res (numpy array): Healthy volume.
        region_to_place_tumour_mask (numpy array): Mask of the region to where the tumour can be placed.
        segmentation (numpy array): Tumour segmentation.
    """
    # Padding the volume so no region ouside of the volume is selected
    healthy_ct_scan_full_res = np.pad(healthy_ct_scan_full_res, pad_width=64, mode='constant', constant_values=-200) # -200 background
    region_to_place_tumour_mask = np.pad(region_to_place_tumour_mask, pad_width=64, mode='constant', constant_values=1) # 1 means that the tumour cannot be placed there
    
    # Select a random center
    voxel_indices = np.argwhere(region_to_place_tumour_mask == 2)
    random_voxel_indices = voxel_indices[np.random.choice(len(voxel_indices), size=1, replace=False)]
    random_x, random_y, random_z = random_voxel_indices[0][0], random_voxel_indices[0][1], random_voxel_indices[0][2]

    # Crop the full resolution scan and mask
    healthy_ct_scan = healthy_ct_scan_full_res[
        random_x-64:random_x+64,
        random_y-64:random_y+64,
        random_z-64:random_z+64
        ]
    region_to_place_tumour_mask_crop = region_to_place_tumour_mask[ 
        random_x-64:random_x+64,
        random_y-64:random_y+64,
        random_z-64:random_z+64
        ] 
    # Ensure the segmentation remains within the anatomical boundaries defined by the region_to_place_tumour_mask_crop
    segmentation[region_to_place_tumour_mask_crop == 1] = 0 

    # Keep the original intensities of the cropped region
    healthy_ct_scan_origin_intensities = np.copy(healthy_ct_scan)
    
    # Convert to torch and add two dimentions
    healthy_ct_scan = th.from_numpy(healthy_ct_scan)
    healthy_ct_scan = rescale_array(healthy_ct_scan, minv=-1, maxv=1)
    healthy_ct_scan = healthy_ct_scan.unsqueeze(0).unsqueeze(0).to(device)
    
    healthy_ct_scan_origin_intensities = th.from_numpy(healthy_ct_scan_origin_intensities)
    healthy_ct_scan_origin_intensities = healthy_ct_scan_origin_intensities.unsqueeze(0).unsqueeze(0).to(device)
    segmentation = th.from_numpy(segmentation)
    segmentation = segmentation.unsqueeze(0).unsqueeze(0).to(device)
    healthy_ct_scan_full_res = th.from_numpy(healthy_ct_scan_full_res)
    healthy_ct_scan_full_res = healthy_ct_scan_full_res.unsqueeze(0).unsqueeze(0).to(device)
    
    return healthy_ct_scan_full_res, healthy_ct_scan, healthy_ct_scan_origin_intensities, segmentation, random_voxel_indices

def get_affine_and_header(file_path):
  """
  Extracts the affine transformation matrix and header information from a NIfTI file.
  Args:
    filename (str): The path to the NIfTI file.
  Returns:
    tuple: A tuple containing the affine matrix and header information.
  """
  img = nib.load(file_path)
  affine = img.affine
  header = img.header
  return affine, header

def get_list_of_segs(seg_path, json_file):
    # Get list of segmentations
    seg_path = "../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/seg"
    with open(json_file, 'r') as file:
        data_train = json.load(file)
        data_train = data_train['training']
    allowed_segs = []
    for case in data_train:
        seg_file = case['seg'].replace('seg/','')
        allowed_segs.append(seg_file)

    segs_list = []
    for case in os.listdir(seg_path):
        if "empty" not in case:
            if case in allowed_segs:
                segs_list.append(case)
    print(f"segs_list cases: {len(segs_list)}")
    return segs_list

def main():
    args = create_argparser().parse_args()

    # Reverse wavelet transform
    idwt = IDWT_3D("haar")
    dwt = DWT_3D('haar')

    # Set device
    args.devices = [th.cuda.current_device()]
    dist_util.setup_dist(devices=args.devices)

    # Set logger
    logger.configure()
    logger.log("Creating model and diffusion...")

    # Handling arguments to ensure they are Boolean
    args.use_wavelet = True if (args.use_wavelet=="True" or args.use_wavelet==True) else False
    args.use_label_cond_conv = True if (args.use_label_cond_conv=="True" or args.use_label_cond_conv==True) else False
    args.use_dilation = True if (args.use_dilation=="True" or args.use_dilation==True) else False
    args.use_data_augmentation = True if (args.use_data_augmentation=="True" or args.use_data_augmentation==True) else False
    args.no_seg = True if (args.no_seg=="True" or args.no_seg==True) else False
    args.ROI_DataAug = True if (args.ROI_DataAug=="True" or args.ROI_DataAug==True) else False
    args.from_monai_loader = True if (args.from_monai_loader=="True" or args.from_monai_loader==True) else False

    print(f"Doing inference with arguments: {args}")

    # Set model (U-net) and scheduler (Gaussian)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    model.eval()

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    # Loading the dataset for the conditional generation
    if args.dataset == 'brats':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        ds = BRATSVolumes(args.data_dir, test_flag=False,
                          normalize=(lambda x: 2*x - 1) if args.renormalize else None,
                          mode='train',
                          img_size=args.image_size)
    elif args.dataset == 'lidc-idri':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        ds = LIDCVolumes(args.data_dir, test_flag=False,
                         normalize=(lambda x: 2*x - 1) if args.renormalize else None,
                         mode='train',
                         img_size=args.image_size)
    elif args.dataset == 'hnn':
        from_monai_loader = True
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        dl, ds = HnNVolumes(
            args=args,
            directory=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize=(lambda x: 2*x - 1) if args.renormalize else None,
            mode='test',
            img_size=args.image_size,
            no_seg=args.no_seg,
            full_background=args.full_background,
            clip_min=int(args.clip_min),
            clip_max=int(args.clip_max)).get_dl_ds()
    
    elif args.dataset == 'c_brats':
        from_monai_loader = True
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        dl, ds = c_BraTSVolumes(
            directory=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_key=args.modality,
            normalize=(lambda x: 2*x - 1) if args.renormalize else None,
            mode='test',
            img_size=args.image_size,
            no_seg=args.no_seg).get_dl_ds()
        if args.full_background:
            warnings.warn("full_background set to True but it is not used in the BraTS dataset", UserWarning)
    
    elif args.dataset == 'hnn_tumour_inpainting':
        if args.from_monai_loader == False:
            with open('../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json', 'r') as file:
                training_cases = json.load(file)
                training_cases = training_cases['training']
            dl = training_cases # Synthtic cases TODO

            #dl = []
            #for case in training_cases: # TODO For real cases
            #    if "empty" not in case['seg']:
            #        dl.append(case)

            # TODO Synthtic cases
            segs_list = get_list_of_segs(seg_path='../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/seg', 
                                         json_file='../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json'
                                         )
            from_monai_loader = False
        else:
            print(f"Keep in mind that some cases are missing because they have tumours bigger than 128x128x128")
            args.csv_path = args.data_dir
            dl = data_inpaint_utils.get_loader(args) 
            from_monai_loader = True
    else:
        print("We currently just support the datasets: brats, lidc-idri, hnn, c_brats, hnn_tumour_inpainting")
    
    if args.dataset == 'hnn' or args.dataset == 'c_brats' or args.dataset == 'hnn_tumour_inpainting':
        datal = dl
    else:
        datal = th.utils.data.DataLoader(ds,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=True,
                                        )
    # Creation of the output directory
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"The inferences will be saved in: {args.output_dir}")

    # Start iteration through data loader
    for ind, batch in enumerate(datal):
        if from_monai_loader==False: 
            case_name = f"{batch['image'].split('/')[-1].split('.nii.gz')[0]}"
            
            #case_path = os.path.join(args.input_dir, batch['image']) # Original data  TODO
            #seg_path = os.path.join(args.input_dir, batch['seg']) # Original data TODO: to get the correspondent segmentation

            case_path = os.path.join(args.input_dir, f"{case_name}_CT_n0.nii.gz") # TODO synthetic data
            seg_name = random.choice(segs_list) # TODO synthetic data
            seg_path = os.path.join('../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/seg', seg_name) # TODO synthetic data
            
            mask_file_path = os.path.join(f"./results/Synthetic_Datasets/Whole_scans/Bone_segmentation/Mask_for_tumour_inpaint", args.output_dir.split('/')[-2], "Original_1000", f"{case_name}_CT_n0_tumour_place.nii.gz")
            contrast_value = batch["contrast"]
        else:
            # Each batch contains the same data used for training, i.e., scans and conditions
            if "scan_ct_meta_dict" in batch:
                case_path = batch['scan_ct_meta_dict']['filename_or_obj'][0]
                seg_path = batch['label_meta_dict']['filename_or_obj'][0]
            elif "image_meta_dict" in batch:
                case_path = batch['image_meta_dict']['filename_or_obj'][0]
                seg_path = batch['seg_meta_dict']['filename_or_obj'][0]
            elif "t1c_meta_dict" in batch:
                case_path = batch['t1c_meta_dict']['filename_or_obj'][0]
                seg_path = batch['seg_meta_dict']['filename_or_obj'][0]
            print(f"Loaded {case_path}")
            print(f"Loaded {seg_path}")
            # Set case id
            case_name = case_path.split('/')[-1].split(".nii.gz")[0]
        #try:
        if os.path.isfile(os.path.join(args.output_dir, f'{case_name}_CT_n0.nii.gz')):
            continue # Check if file was generated already
        if "empty" in seg_path and args.no_seg==False and (args.train_mode!="default_tumour_inpainting" or args.train_mode!="tumour_inpainting"):
            print(f"Skipping: {seg_path}")
            pass
        else:
            # Defining the img (noise) input to the model
            if args.use_wavelet:
                wavelet_coefficients = int(len(args.modality.split("_"))*8)
                img = th.randn(args.batch_size,         # Batch size
                            wavelet_coefficients,    # 8 wavelet coefficients
                            args.image_size//2,      # Half spatial resolution (D)
                            args.image_size//2,      # Half spatial resolution (H)
                            args.image_size//2,      # Half spatial resolution (W)
                            ).to(dist_util.dev())
            else:
                img = th.randn(args.batch_size,         # Batch size
                                1,    
                                args.image_size,      # Half spatial resolution (D)
                                args.image_size,      # Half spatial resolution (H)
                                args.image_size,      # Half spatial resolution (W)
                                ).to(dist_util.dev())

            model_kwargs = {} # Not really used
            affine, header = get_affine_and_header(case_path)

            if args.dataset == 'hnn_tumour_inpainting':
                # Setting conditions and healthy scan for tumour inpainting
                if from_monai_loader:
                    # Using the data coming from the MONAI data loader
                    # This contains real scans, real segmentations and the real contrast. 
                    # For fake scans set  from_monai_loader to False
                    scan_ct = batch["scan_ct"].to(dist_util.dev())
                    healthy_ct_scan_full_res = batch["scan_ct_origin"].to(dist_util.dev())
                    healthy_ct_scan_origin_intensities = th.clone(healthy_ct_scan_full_res)
                    original_healthy_tensor = batch["scan_volume_crop_pad"].to(dist_util.dev())
                    label = batch["label"].to(dist_util.dev())
                    label_crop_pad = batch["label_crop_pad"].to(dist_util.dev())
                    contrast = batch["contrast"].to(dist_util.dev())
                    contrast_tensor = batch["contrast_tensor"].to(dist_util.dev())
                    no_contrast_tensor = batch["no_contrast_tensor"].to(dist_util.dev())
                    if args.train_mode == 'tumour_inpainting':
                        healthy_ct_scan = original_healthy_tensor
                        label_crop_pad_dillated = label_crop_pad.cpu().detach().clone().numpy().squeeze()
                        structuring_element = np.ones((3, 3, 3), dtype=str)
                        dilated_mask = binary_dilation(label_crop_pad_dillated, structure=structuring_element, iterations=5)
                        dilated_mask = th.from_numpy(dilated_mask).float()
                        dilated_mask = dilated_mask.unsqueeze(dim=0).unsqueeze(dim=0).cuda()
                        blured_mask = blur_mask_3d(mask=dilated_mask, label_crop_pad=label_crop_pad, blur_factor=25, blur_type=args.blur_mask)
                    else:
                        healthy_ct_scan = th.clone(no_contrast_tensor)
                else:
                    if case_path.endswith("_CT_n0.nii.gz"):
                        print("Random tumour")
                        print(f"Number of cases for inference: {len(datal)}")
                        print(f"Number of segmentations: {len(segs_list)}")
                        sagittal, coronal, axial, no_contrast_tensor, contrast_tensor, label_crop_pad, healthy_ct_scan, healthy_ct_scan_origin_intensities, healthy_ct_scan_full_res = get_label_condition(args=args, case_path=case_path, contrast_value=contrast_value, seg_path=seg_path, mask_file_path=mask_file_path)
                        print(f"sagittal: {sagittal}")
                        print(f"coronal: {coronal}")
                        print(f"axial: {axial}")
                    else:
                        print("Tumour in the same position")
                        print(f"Number of cases for inference: {len(datal)}")
                        sagittal, coronal, axial, no_contrast_tensor, contrast_tensor, label_crop_pad, healthy_ct_scan, healthy_ct_scan_origin_intensities, healthy_ct_scan_full_res = get_label_condition_no_random(args=args, case_path=case_path, contrast_value=contrast_value, seg_path=seg_path)
                        print(f"sagittal: {sagittal}")
                        print(f"coronal: {coronal}")
                        print(f"axial: {axial}")
                    if args.train_mode == 'tumour_inpainting':
                        label_crop_pad_dillated = label_crop_pad.cpu().detach().clone().numpy().squeeze()
                        structuring_element = np.ones((3, 3, 3), dtype=str)
                        dilated_mask = binary_dilation(label_crop_pad_dillated, structure=structuring_element, iterations=5)
                        dilated_mask = th.from_numpy(dilated_mask).float()
                        dilated_mask = dilated_mask.unsqueeze(dim=0).unsqueeze(dim=0).cuda()
                        blured_mask = blur_mask_3d(dilated_mask, label_crop_pad, blur_factor=25, blur_type=args.blur_mask) 
                    else:
                        healthy_ct_scan = th.clone(no_contrast_tensor)

                # Setting the label_condition
                if args.blur_mask=="edge_blur" or args.blur_mask=="full_blur":
                    print(f"USING BLUR MASK: {args.blur_mask}")
                    label_condition = th.cat((no_contrast_tensor, contrast_tensor, label_crop_pad, blured_mask, healthy_ct_scan), dim=1)
                    
                else:
                    label_condition = th.cat((no_contrast_tensor, contrast_tensor, label_crop_pad, healthy_ct_scan), dim=1)
            
            elif args.dataset == 'hnn' and args.train_mode == 'concat_cond':
                label_condition = batch["seg"].to(dist_util.dev())
                if label_condition.shape[1]==2:
                    segmentation = th.zeros_like(label_condition[:,0:1,:,:,:])
                else:
                    segmentation = label_condition[:,2:3,:,:,:] 

                contrast_tensor = label_condition[0][1]
                no_contrast_tensor = label_condition[0][0]
                resize = Resize((128, 128, 128), size_mode='all', mode="nearest", align_corners=None, anti_aliasing=False, anti_aliasing_sigma=None, dtype=th.float32, lazy=False)
                label_condition = resize(label_condition[0]).unsqueeze(0)
            elif args.dataset == 'c_brats':
                label_condition = batch['seg'].to(dist_util.dev())
                # Convert the label to the BraTS format
                tumour_core = label_condition[0][0]
                whole_tumour = label_condition[0][1]
                enhancing_tumour = label_condition[0][2]
                segmentation = th.zeros_like(tumour_core)
                segmentation[whole_tumour==1] = 2
                segmentation[tumour_core==1] = 1
                segmentation[enhancing_tumour==1] = 3
                # Crop segmentation
                segmentation = segmentation.cpu().numpy()
                segmentation = segmentation[8:-8, 8:-8, 50:-51]
                # Save segmentation
                output_name = os.path.join(args.output_dir, f'{case_name[:-4]}_label_n0.nii.gz') # TODO
                segmentation = np.flip(segmentation, axis=1) 
                segmentation = np.flip(segmentation, axis=0) 
                segmentation_nii = nib.Nifti1Image(segmentation, affine=affine, header=header) 
                nib.save(img=segmentation_nii, filename=output_name)
                print(f'Saved to {output_name}')
                
                if args.train_mode == 'concat_cond':
                    resize = Resize((128, 128, 128), size_mode='all', mode="nearest", align_corners=None, anti_aliasing=False, anti_aliasing_sigma=None, dtype=th.float32, lazy=False)
                    label_condition = resize(label_condition[0]).unsqueeze(0)
            else:
                label_condition = batch['seg'].cuda()
                contrast_tensor = label_condition[0][1]
                no_contrast_tensor = label_condition[0][0]

                if label_condition.shape[1]==2:
                    segmentation = th.zeros_like(label_condition[:,0:1,:,:,:])
                else:
                    segmentation = label_condition[:,2:3,:,:,:] #
            
            if args.train_mode == 'wavelet_cond':
                LLL = None
                for condition in label_condition[0]:
                    condition = condition.unsqueeze(0).unsqueeze(0)
                    if LLL==None:
                        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(condition)
                        cond_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
                    else:
                        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(condition)
                        cond_dwt = th.cat([cond_dwt, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
                label_condition = cond_dwt

            if (args.train_mode=="concat_cond" or args.train_mode=="conv_before_concat" or args.train_mode=="wavelet_cond") and args.dataset!='c_brats':
                # Check if the scan has contrast or not
                if th.sum(contrast_tensor) != 0:
                    cube_coords = th.nonzero(contrast_tensor) 
                    min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                    max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                else:
                    cube_coords = th.nonzero(no_contrast_tensor) 
                    min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                    max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                
                # Cropping the output of the model considering the ROI
                x_min, y_min, z_min = min_coords
                x_max, y_max, z_max = max_coords
                segmentation = segmentation[:, :, x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                contrast_tensor = contrast_tensor[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                no_contrast_tensor = no_contrast_tensor[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        

                if th.sum(segmentation)!=0:
                    print("In segmentation")
                    output_name = os.path.join(args.output_dir, f'{case_name}_label_n0.nii.gz') # TODO
                    segmentation_nii = nib.Nifti1Image(np.flip(segmentation.cpu().numpy()[0][0], axis=1), affine=affine, header=header) 
                    nib.save(img=segmentation_nii, filename=output_name)
                    print(f'Saved to {output_name}')

            # Defining sampling loop
            sample_fn = diffusion.p_sample_loop 
            model_output, label_condition = sample_fn(model=model,
                            shape=img.shape,
                            noise=img,
                            time=args.sampling_steps,
                            mode=args.train_mode,
                            label_condition=label_condition,
                            use_wavelet=args.use_wavelet,
                            use_label_cond_conv=args.use_label_cond_conv,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=None,
                                )

            if args.use_wavelet:
                B, C, D, H, W = model_output.size()

                # Convert output of the model back to full resolution 
                modal_idx = -1
                for model_output_idx in range(0, C, 8):
                    modal_idx +=1
                    sample = idwt(model_output[:, 0+modal_idx*8, :, :, :].view(B, 1, H, W, D) * 3.,
                                        model_output[:, 1+modal_idx*8, :, :, :].view(B, 1, H, W, D),
                                        model_output[:, 2+modal_idx*8, :, :, :].view(B, 1, H, W, D),
                                        model_output[:, 3+modal_idx*8, :, :, :].view(B, 1, H, W, D),
                                        model_output[:, 4+modal_idx*8, :, :, :].view(B, 1, H, W, D),
                                        model_output[:, 5+modal_idx*8, :, :, :].view(B, 1, H, W, D),
                                        model_output[:, 6+modal_idx*8, :, :, :].view(B, 1, H, W, D),
                                        model_output[:, 7+modal_idx*8, :, :, :].view(B, 1, H, W, D))
            else:
                sample = model_output

            if len(sample.shape) == 5:
                sample = sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1
            
            
            modal_idx = 0
            sample_denorm = np.clip(sample.detach().cpu().numpy()[0, :, :, :], a_min=-1, a_max=1) # remove very high and low values

            if args.train_mode == "tumour_inpainting":
                roi = sample * blured_mask[0] 
                not_roi = healthy_ct_scan * (1-blured_mask[0])
                sample = roi + not_roi

                sample_denorm = np.clip(sample.detach().cpu().numpy()[0, :, :, :], a_min=-1, a_max=1) # remove very high and low values
                
                sample_denorm = rescale_array(
                    arr=sample_denorm, 
                    minv=(healthy_ct_scan_origin_intensities.cpu().numpy().min()), 
                    maxv=(healthy_ct_scan_origin_intensities.cpu().numpy().max())
                    )
            elif args.dataset=='c_brats':
                data = nib.load(case_path).get_fdata()
                out_clipped = np.clip(data, np.quantile(data, 0.001), np.quantile(data, 0.999))
                clip_min = np.min(out_clipped)
                clip_max = np.max(out_clipped)

                sample_denorm = rescale_array(
                        arr=sample_denorm, 
                        minv=int(clip_min), 
                        maxv=int(clip_max)
                        )
            else:
                sample_denorm = rescale_array(
                    arr=sample_denorm, 
                    minv=int(args.clip_min), 
                    maxv=int(args.clip_max)
                    )
            
            if args.train_mode=="default_tumour_inpainting":
                sample_denorm_corrected = np.flip(sample_denorm, axis=1)  
                synth_ct_scan_output = os.path.join(args.output_dir, f'{case_name}_CT_n0.nii.gz') # TODO
                # Saving the output of the model
                img = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header) 
                nib.save(img=img, filename=synth_ct_scan_output)
                # Saving the label
                output_name = os.path.join(args.output_dir, f'{case_name}_label_n0.nii.gz') # TODO
                label_corrected = np.flip(label_crop_pad[0][0].detach().cpu().numpy(), axis=1)  
                img = nib.Nifti1Image(label_corrected, affine=affine, header=header)
                nib.save(img=img, filename=output_name)
                print(f'Output of the model saved to {synth_ct_scan_output}')
            elif args.dataset == 'c_brats':
                sample_denorm_corrected = sample_denorm[8:-8, 8:-8, 50:-51]
                sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1) 
                sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=0) 
                # Saving the output of the model
                synth_ct_scan_output = os.path.join(args.output_dir, f'{case_name}_n0.nii.gz') # TODO
                img = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header) 
                nib.save(img=img, filename=synth_ct_scan_output)
                print(f'Output of the model saved to {synth_ct_scan_output}')
            elif args.train_mode == "tumour_inpainting":
                # Save the full resolution scan with the synthetic tumour
                new_ct_scan_full_res_np =  np.copy(healthy_ct_scan_full_res.detach().cpu().numpy())
                new_ct_scan_full_res_np = new_ct_scan_full_res_np[0][0]
                new_ct_scan_full_res_np[sagittal[0]:sagittal[1], coronal[0]:coronal[1], axial[0]:axial[1]] = sample_denorm
                output_name = os.path.join(args.output_dir, f'{case_name}_CT_n0.nii.gz')
                img = nib.Nifti1Image(new_ct_scan_full_res_np[64:-64, 64:-64, 64:-64,], affine=affine, header=header)
                nib.save(img=img, filename=output_name)
                # 
                new_label_full_res_np = np.zeros_like(new_ct_scan_full_res_np)
                new_label_full_res_np[sagittal[0]:sagittal[1], coronal[0]:coronal[1], axial[0]:axial[1]] = label_crop_pad[0][0].float().detach().cpu().numpy()
                output_name = os.path.join(args.output_dir, f'{case_name}_label_n0.nii.gz')
                img = nib.Nifti1Image(new_label_full_res_np[64:-64, 64:-64, 64:-64,], affine=affine, header=header) 
                nib.save(img=img, filename=output_name)
            else:
                sample_denorm_corrected = sample_denorm[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] 
                # Saving the output of the model
                sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1)  
                synth_ct_scan_output = os.path.join(args.output_dir, f'{case_name}_CT_n0.nii.gz') # TODO
                img = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header) 
                nib.save(img=img, filename=synth_ct_scan_output)
                print(f'Output of the model saved to {synth_ct_scan_output}')
                    
        #except Exception as e:
        #    print(f"Error -> case_path: {case_path}") 
        #    print(f"Error -> seg_path: {seg_path}") 
        #    print(f"{e}") 
            

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=None,
        output_dir='./results',
        mode='c_sample',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        num_workers=8,
        full_background=False,
        modality=None,
        clip_min=None,
        clip_max=None,
        use_label_cond=None,
        use_wavelet=None,
        use_dilation=None, 
        use_data_augmentation=None, 
        train_mode=None,
        ROI_DataAug=None,
        no_seg=None,
        blur_mask=None,
        from_monai_loader=None,
        input_dir=None,
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()