# This script generates a dataset with cropped cases (not full resolution)
import sys
sys.path.append("..")
import os
from monai.transforms import Compose, LoadImage, CropForeground, EnsureChannelFirst, ResizeWithPadOrCrop, ScaleIntensityRange
from guided_diffusion.c_unet import SuperResModel, UNetModel, EncoderUNetModel
import torch
import torch as th
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
import nibabel as nib
import numpy as np
idwt = IDWT_3D("haar")
dwt = DWT_3D("haar")
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureTyped,
    ScaleIntensityRanged,
    CopyItemsd,
    EnsureChannelFirstd,
    CopyItemsd,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandSpatialCropd,
    Lambda
    )
from tqdm import tqdm
import torch
from scipy.ndimage import center_of_mass
from monai.data import CSVDataset
from monai.data.utils import pad_list_data_collate
from utils.convert_head_n_neck_cancer import ConvertHeadNNeckCancerd as LABEL_TRANSFORM
from utils.crop_scan_center_in_tumour import CropScanCenterInTumour

def get_tensor(file_path, norm, clip):
    """
    Loads the nii.gz file, and normalises if necessary.
    Arguments:
        file_path (str): Path to the nii.gz file.
        norm (bool): True for clipping and normalisation.
    Return:
        Numpy array of nii.gz file.
    """
    transforms = [
        LoadImage(image_only=True),
        EnsureChannelFirst()
        ]
    if clip:
        transforms.append(
        ScaleIntensityRange(a_min=-200, a_max=200, b_min=-200, b_max=200, clip=True)
        )
    if norm:
        transforms.append(
        ScaleIntensityRange(a_min=-200, a_max=200, b_min=-1, b_max=1, clip=True)
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

def get_model(in_channels=11, out_channels=8, channel_mult=[1, 2, 2, 4, 4, 4], label_cond_in_channels=0, use_label_cond_conv=False, pretrained_weights_path=None):
    model = UNetModel(
        image_size=128,
        in_channels=in_channels,
        model_channels=64,
        out_channels=out_channels,
        num_res_blocks=2,
        attention_resolutions=tuple([]),
        dropout=0.0,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=True,
        use_new_attention_order=False,
        dims=3,
        num_groups=32,
        bottleneck_attention=False,
        additive_skips=True,
        resample_2d=False,
        label_cond_in_channels=label_cond_in_channels,
        use_label_cond_conv=use_label_cond_conv,
    )
    # Load the pre-trained weights
    state_dict = torch.load(pretrained_weights_path, map_location=torch.device('cuda:0'))  # Load to CPU, or adjust for GPU if needed

    # Load weights into the model
    model.load_state_dict(state_dict)

    return model

def get_scheduler(sch, num_inference_steps):
    if sch=="DPM++_2M":
        use_karras_sigmas = False
        algorithm_type = "dpmsolver++"
    elif sch=="DPM++_2M_Karras":
        use_karras_sigmas = True
        algorithm_type = "dpmsolver++"
    elif sch=="DPM++_2M_SDE":
        use_karras_sigmas = False
        algorithm_type = "sde-dpmsolver++"
    elif sch=="DPM++_2M_SDE_Karras":
        use_karras_sigmas = True
        algorithm_type = "sde-dpmsolver++"
        
    scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000, 
            variance_type="fixed_large", 
            prediction_type="sample", 
            use_karras_sigmas=use_karras_sigmas, 
            algorithm_type=algorithm_type
            #use_beta_sigmas=True # https://huggingface.co/papers/2407.12173
            )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    return scheduler

#---
# Cropped CT scans - Inpainting - random label crop 
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

def get_loader(CSV_PATH, NUM_WORKERS, use_data_augmentation, clip_min, clip_max): 
    
    scan_name = "scan_ct"
    col_names = ['scan_ct', 'label', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size', 'contrast']
    col_types= {'center_x': {'type': int}, 'center_y': {'type': int}, 'center_z': {'type': int}, 'x_extreme_min': {'type': int}, 'x_extreme_max': {'type': int}, 'y_extreme_min': {'type': int}, 'y_extreme_max': {'type': int}, 'z_extreme_min': {'type': int}, 'z_extreme_max': {'type': int}, 'x_size': {'type': int}, 'y_size': {'type': int}, 'z_size': {'type': int}, 'contrast': {'type': int}}      
    print(f"Scan Modality: {scan_name}")

    train_transforms = [
                    LoadImaged(keys=[scan_name, 'label'], image_only=False),
                    EnsureChannelFirstd(keys=[scan_name, "label"]),
                    EnsureTyped(keys=[scan_name, "label"]),
                    CopyItemsd(keys=[scan_name], names=[f"{scan_name}_origin"]),
                    ScaleIntensityRanged(keys=[f"{scan_name}_origin"], a_min=int(clip_min), a_max=int(clip_max), b_min=int(clip_min), b_max=int(clip_max), clip=True),
                    LABEL_TRANSFORM(keys="label"),
                ] 

    new_keys = ['scan_volume_crop_pad', 'label_crop_pad']
    interpolation_mode=['trilinear', 'nearest']

    if use_data_augmentation:
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
            ScaleIntensityRanged(keys=[scan_name, "scan_volume_crop_pad"], a_min=int(clip_min), a_max=int(clip_max), b_min=-1.0, b_max=1.0, clip=True)
        )
        train_transforms.append(
            ToTensord(keys=[scan_name, 'no_contrast_tensor', 'contrast_tensor', 'scan_volume_crop_pad', 'label', 'label_crop_pad'])
            )
    else:
        train_transforms.append(
            CropScanCenterInTumour(keys=scan_name, dilation=False, translate_range=None)
            )       
        train_transforms.append(
            ScaleIntensityRanged(keys=[scan_name], a_min=int(clip_min), a_max=int(clip_max), b_min=-1.0, b_max=1.0, clip=True)
        )
        
        train_transforms.append(ToTensord(keys=[scan_name, 'no_contrast_tensor', 'contrast_tensor', 'scan_volume_crop', 'scan_volume_crop_pad', 'label', 'label_crop_pad', 'center_x', 'center_y', 'center_z', 'x_extreme_min', 'x_extreme_max', 'y_extreme_min', 'y_extreme_max', 'z_extreme_min', 'z_extreme_max', 'x_size', 'y_size', 'z_size']))
        
    final_train_transforms = Compose(train_transforms)
    
    train_CSVdataset = CSVDataset(src=CSV_PATH, col_names=col_names, col_types=col_types)
    train_CSVdataset = CacheDataset(train_CSVdataset, transform=final_train_transforms, cache_rate=0, num_workers=NUM_WORKERS, progress=True)  
    train_loader = DataLoader(train_CSVdataset, batch_size=int(1), num_workers=NUM_WORKERS, drop_last=True, shuffle=False, collate_fn=pad_list_data_collate)
    
    print(f"Number of training images: {len(train_CSVdataset)}")
    print(f'Dataset training: number of batches: {len(train_loader)}')
    print("Leaving the data loader. Good luck!") 
    return train_loader

#### hnn_tumour_inpainting_CT_default_tumour_inpainting__data_augment_20_11_2024_11:07:31
# * HU between -200 and 200. tumour weight 10.

def run_inference(train_loader, model, scheduler_list, n, num_inference_steps, clip_min, clip_max, root_output_path, cycle_id):
    model.cuda()
    for idx, batch  in enumerate(train_loader):
        print(f"Case number {idx}")
        if "scan_ct_meta_dict" in batch:
            case_path = batch['scan_ct_meta_dict']['filename_or_obj'][0]
        elif "image_meta_dict" in batch:
            case_path = batch['image_meta_dict']['filename_or_obj'][0]
        print(f"Loaded {case_path}")
        
        # Set case id
        case_name = case_path.split('/')[-1].split(".nii.gz")[0]
        for sch in scheduler_list:
            out_path = os.path.join(root_output_path, sch)
            scheduler = get_scheduler(sch, num_inference_steps)
            noise_start = torch.randn(1, 1, 128, 128, 128)  
            # Prepare the noisy image
            final_scan = noise_start.clone().detach()
            final_scan = final_scan.cuda()

            segmentation = batch["label_crop_pad"].cuda()
            no_contrast_tensor = batch["no_contrast_tensor"].cuda()
            contrast_tensor = batch["contrast_tensor"].cuda()
            label_condition = torch.cat((no_contrast_tensor, contrast_tensor, segmentation), dim=1)

            input_model = torch.cat((final_scan, label_condition), dim=1)
            input_model = input_model.cuda()

            affine, header = get_affine_and_header(case_path)
            segmentation = np.flip(segmentation.cpu().numpy().astype(float)[0][0], axis=1)
            nii_image = nib.Nifti1Image(segmentation, affine=affine, header=header) 
            seg_ct_scan_output = os.path.join(out_path, f'{case_name}_label_n{cycle_id}.nii.gz')
            nib.save(nii_image, seg_ct_scan_output)

            # Start the reverse process (denoising from noise)
            for timestep in tqdm(scheduler.timesteps, desc="Processing timesteps"):
                # Get the current timestep's noise
                t = torch.tensor([timestep] * final_scan.shape[0])
                t = t.cuda()
                # Perform one step of denoising
                with torch.no_grad():
                    model_kwargs = {}
                    noise_pred = model(input_model, timesteps=t, label_condition=label_condition, **model_kwargs)
                    # Update the noisy_latents (reverse the noise process)
                    final_scan = scheduler.step(model_output=noise_pred, timestep=timestep, sample=final_scan)
                    final_scan = final_scan['prev_sample']
                    input_model = torch.cat((final_scan, label_condition), dim=1)

            # Assuming final_image is a PyTorch tensor
            # Convert the final_image tensor to a NumPy array if it's a tensor
            final_image_np = final_scan.squeeze().cpu().numpy()  # Remove the channel dim and move to CPU
            synth_ct_scan_output = os.path.join(out_path, f'{case_name}_CT_n{cycle_id}.nii.gz')  
            sample_denorm = np.clip(final_image_np, a_min=-1, a_max=1) # remove very high and low values

            sample_denorm = rescale_array(
                            arr=sample_denorm, 
                            minv=int(clip_min), 
                            maxv=int(clip_max)
                            )

            sample_denorm_corrected = sample_denorm
            sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1)
            nii_image = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header)  
            nib.save(nii_image, synth_ct_scan_output)

            
        if idx+1 == n:
            break
            
def tumour_inpainting_200(number_of_cylces):
    for cycle_id in range(number_of_cylces):
        clip_min = -200
        clip_max = 200
        in_channels = 4
        out_channels = 1
        label_cond_in_channels = 0
        use_label_cond_conv = False
        pretrained_weights_path = '../runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_6_5_2025_12:31:43/checkpoints/hnn_tumour_inpainting_2000000.pt'  # Specify the correct path
        channel_mult=[1, 2, 2, 4, 4]
        model = get_model(in_channels=in_channels, 
                        out_channels=out_channels,
                        channel_mult=channel_mult,
                        label_cond_in_channels=label_cond_in_channels, 
                        use_label_cond_conv=use_label_cond_conv,
                        pretrained_weights_path=pretrained_weights_path)
        model.eval()
        model.cuda()

        NUM_WORKERS = 4
        CSV_PATH = "../utils/hnn.csv" # cases with no empty label not considered
        use_data_augmentation = True
        train_loader = get_loader(CSV_PATH, NUM_WORKERS, use_data_augmentation, clip_min, clip_max)
        print("Loaded model and data loader")

        # Control inference parameters
        scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]
        root_output_path = "../results/Synthetic_Datasets/Cropped/200/"

        for sch in scheduler_list:
            os.makedirs(os.path.join(root_output_path, sch), exist_ok=True)

        n = len(train_loader)
        print(f"Doing {n} cases")
        num_inference_steps = 100

        run_inference(train_loader=train_loader,
                    model=model,
                    scheduler_list=scheduler_list,
                    n=n, 
                    num_inference_steps=num_inference_steps, 
                    clip_min=clip_min,
                    clip_max=clip_max, 
                    root_output_path=root_output_path,
                    cycle_id=cycle_id)
        

#### hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_28_11_2024_14:37:59
#* HU between -1000 and 1000. tumour weight 10. 
def run_inference(train_loader, model, scheduler_list, n, num_inference_steps, clip_min, clip_max, root_output_path, cycle_id):
    model.cuda()
    for idx, batch  in enumerate(train_loader):
        print(f"Case number {idx}")
        if "scan_ct_meta_dict" in batch:
            case_path = batch['scan_ct_meta_dict']['filename_or_obj'][0]
        elif "image_meta_dict" in batch:
            case_path = batch['image_meta_dict']['filename_or_obj'][0]
        print(f"Loaded {case_path}")
        # Set case id
        case_name = case_path.split('/')[-1].split(".nii.gz")[0]

        for sch in scheduler_list:
            scheduler = get_scheduler(sch, num_inference_steps)
            out_path = os.path.join(root_output_path, sch)
            noise_start = torch.randn(1, 1, 128, 128, 128)  
            # Prepare the noisy image
            final_scan = noise_start.clone().detach()
            final_scan = final_scan.cuda()

            segmentation = batch["label_crop_pad"].cuda()
            no_contrast_tensor = batch["no_contrast_tensor"].cuda()
            contrast_tensor = batch["contrast_tensor"].cuda()
            label_condition = torch.cat((no_contrast_tensor, contrast_tensor, segmentation), dim=1)

            input_model = torch.cat((final_scan, label_condition), dim=1)
            input_model = input_model.cuda()

            affine, header = get_affine_and_header(case_path)
            #segmentation = np.flip(segmentation.cpu().numpy().astype(float)[0][0], axis=1)
            nii_image = nib.Nifti1Image(segmentation.cpu().numpy().astype(float)[0][0], affine=affine, header=header) 
            seg_ct_scan_output = os.path.join(out_path, f'{case_name}_label_n{cycle_id}.nii.gz') 
            nib.save(nii_image, seg_ct_scan_output)

            # Start the reverse process (denoising from noise)
            for timestep in tqdm(scheduler.timesteps, desc="Processing timesteps"):
                # Get the current timestep's noise
                t = torch.tensor([timestep] * final_scan.shape[0])
                t = t.cuda()
                # Perform one step of denoising
                with torch.no_grad():
                    model_kwargs = {}
                    noise_pred = model(input_model, timesteps=t, label_condition=label_condition, **model_kwargs)
                    # Update the noisy_latents (reverse the noise process)
                    final_scan = scheduler.step(model_output=noise_pred, timestep=timestep, sample=final_scan)
                    final_scan = final_scan['prev_sample']
                    input_model = torch.cat((final_scan, label_condition), dim=1)

            # Assuming final_image is a PyTorch tensor
            # Convert the final_image tensor to a NumPy array if it's a tensor
            final_image_np = final_scan.squeeze().cpu().numpy()  # Remove the channel dim and move to CPU
            synth_ct_scan_output = os.path.join(out_path, f'{case_name}_CT_n{cycle_id}.nii.gz')  
            sample_denorm = np.clip(final_image_np, a_min=-1, a_max=1) # remove very high and low values

            sample_denorm = rescale_array(
                            arr=sample_denorm, 
                            minv=int(clip_min), 
                            maxv=int(clip_max)
                            )

            sample_denorm_corrected = sample_denorm
            #sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1)
            nii_image = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header)  
            nib.save(nii_image, synth_ct_scan_output)

        if idx+1 == n:
            break
            
           
def  tumour_inpainting_1000(number_of_cylces):
    for cycle_id in range(number_of_cylces):
        clip_min = -1000
        clip_max = 1000
        in_channels = 4
        out_channels = 1
        label_cond_in_channels = 0
        use_label_cond_conv = False
        pretrained_weights_path = '../runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_6_5_2025_12:32:10/checkpoints/hnn_tumour_inpainting_2000000.pt' # Specify the correct path
        channel_mult=[1, 2, 2, 4, 4]

        model = get_model(in_channels=in_channels, 
                        out_channels=out_channels,
                        channel_mult=channel_mult,
                        label_cond_in_channels=label_cond_in_channels, 
                        use_label_cond_conv=use_label_cond_conv,
                        pretrained_weights_path=pretrained_weights_path)
        model.eval()
        model.cuda()

        NUM_WORKERS = 4
        CSV_PATH = "../utils/hnn.csv" # cases with no empty label not considered
        use_data_augmentation = True
        train_loader = get_loader(CSV_PATH, NUM_WORKERS, use_data_augmentation, clip_min, clip_max)

        print("Loaded model and data loader")

        # Control inference parameters
        scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]
        root_output_path = "../results/Synthetic_Datasets/Cropped/1000/"

        for sch in scheduler_list:
            os.makedirs(os.path.join(root_output_path, sch), exist_ok=True)
        
        n = len(train_loader)
        print(f"Doing {n} cases")
        num_inference_steps = 100

        run_inference(train_loader=train_loader,
                    model=model,
                    scheduler_list=scheduler_list,
                    n=n, 
                    num_inference_steps=num_inference_steps, 
                    clip_min=clip_min,
                    clip_max=clip_max, 
                    root_output_path=root_output_path,
                    cycle_id=cycle_id)
    


#### hnn_tumour_inpainting_CT_default_tumour_inpainting__data_augment_20_11_2024_11:07:31
#* HU between -200 and 200. tumour weight 10.
tumour_inpainting_200(number_of_cylces=1)
 

#### hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_28_11_2024_14:37:59
#* HU between -1000 and 1000. tumour weight 10. 
tumour_inpainting_1000(number_of_cylces=1)
