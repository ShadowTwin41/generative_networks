#########################
#### Unchanged part! ####
#########################
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
from monai.data import load_decathlon_datalist, DataLoader, CacheDataset
from monai.transforms import (
    Compose, 
    LoadImaged,
    EnsureChannelFirstd, 
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged, 
    ResizeWithPadOrCropd,
    CopyItemsd
    )
from utils.data_loader_utils import ConvertToMultiChannel_BackandForeground_Contrastd
from tqdm import tqdm
import torch
from monai.transforms import Resize
from scipy.ndimage import center_of_mass

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

#######################
#### Chaning part! ####
#######################

## Whole CT scan generation ##

def get_loader(image_key, label_key, clip_min, clip_max, image_size, no_seg, full_background, data_split_json, base_dir):
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
    train_transforms.append(EnsureTyped(keys=[image_key, label_key], dtype=torch.float32))
    train_transforms_final =  Compose(train_transforms)

    data_set = load_decathlon_datalist(
                data_split_json,
                is_segmentation=True,
                data_list_key="training",
                base_dir=base_dir,
            )

    print(f"Training cases: {len(data_set)}")

    print(data_set[-1:])
    print(f"TOTAL cases {len(data_set)}")
    # Creating traing dataset
    ds = CacheDataset( 
        data=data_set, 
        transform=train_transforms_final,
        cache_rate=0, 
        copy_cache=False,
        progress=True,
        num_workers=4,
    )
    dl = DataLoader(
                ds,
                batch_size=1,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                shuffle=False, 
                #collate_fn=no_collation,
            )
    return dl, ds, data_set

### Tumour generation ###

#### hnn_CT_conv_before_concat__DA_tumorW_0_28_11_2024_11:19:14 ####
#* HU between -200 and 200. tumour weight 0. DA ROI. ROI and segmentation as condition, feeded first to a conv layer.

def run_inference_conv_before_concat(model, scheduler_list, n, num_inference_steps, clip_min, clip_max, root_output_path, dl, cycle_id):
    model.cuda()

    for idx, batch  in enumerate(dl):
        print(f"Case number {idx}")
        if "scan_ct_meta_dict" in batch:
            case_path = batch['scan_ct_meta_dict']['filename_or_obj'][0]
            seg_path = batch['seg_meta_dict']['filename_or_obj'][0]
        elif "image_meta_dict" in batch:
            case_path = batch['image_meta_dict']['filename_or_obj'][0]
            seg_path = batch['seg_meta_dict']['filename_or_obj'][0]
        print(f"Loaded scan {case_path}")
        print(f"Loaded seg {seg_path}")

        if "empty" not in seg_path:
            affine, header = get_affine_and_header(case_path)
            case_name = case_path.split('/')[-1].split(".nii.gz")[0]

            for sch in scheduler_list:
                out_path = os.path.join(root_output_path, sch)
                scheduler = get_scheduler(sch, num_inference_steps)
                noise_start = torch.randn(1, 8, 128, 128, 128)  
                # Prepare the noisy image
                final_scan = noise_start.clone().detach()
                final_scan = final_scan.cuda()
                input_model = final_scan.cuda()
                label_condition = batch["seg"].cuda()
                segmentation = label_condition[0][2]
                no_contrast_tensor = label_condition[0][0]
                contrast_tensor = label_condition[0][1]

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
                        input_model = final_scan
                B, C, D, H, W = final_scan.size()
                final_scan = idwt(final_scan[:, 0, :, :, :].view(B, 1, H, W, D) * 3.,
                            final_scan[:, 1, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 2, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 3, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 4, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 5, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 6, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 7, :, :, :].view(B, 1, H, W, D))
                # Assuming final_image is a PyTorch tensor
                # Convert the final_image tensor to a NumPy array if it's a tensor
                final_image_np = final_scan.squeeze().cpu().numpy()  # Remove the channel dim and move to CPU

                if th.sum(contrast_tensor) != 0:
                    cube_coords = th.nonzero(contrast_tensor) 
                    min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                    max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                else:
                    cube_coords = th.nonzero(no_contrast_tensor) 
                    min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                    max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                
                sample_denorm = np.clip(final_image_np, a_min=-1, a_max=1) # remove very high and low values

                sample_denorm = rescale_array(
                                arr=sample_denorm, 
                                minv=int(clip_min), 
                                maxv=int(clip_max)
                                )

                # Cropping the output of the model considering the ROI
                x_min, y_min, z_min = min_coords
                x_max, y_max, z_max = max_coords
                sample_denorm_corrected = sample_denorm[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
                segmentation = segmentation[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

                sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1) 
                synth_ct_scan_output = os.path.join(out_path, f'{case_name}_CT_n{cycle_id}.nii.gz')
                nii_image = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header)
                nib.save(nii_image, synth_ct_scan_output)

                segmentation = np.flip(segmentation.cpu().numpy(), axis=1) 
                nii_image = nib.Nifti1Image(segmentation, affine=affine, header=header) 
                seg_ct_scan_output = os.path.join(out_path, f'{case_name}_label_n{cycle_id}.nii.gz')
                nib.save(nii_image, seg_ct_scan_output)
        
        else:
            print(f"{case_path.split('/')[-1].split('.nii.gz')[0]} DONE")

        if idx+1 == n:
            break

def main_conv_before_concat(number_of_cylces):
    for cycle_id in range(number_of_cylces): 
        image_key = "image"
        label_key = "seg"
        image_size = (256, 256, 256)
        data_split_json = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json"
        base_dir = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256"
        full_background = False
        no_seg = False

        clip_min = -200
        clip_max = 200
        in_channels = 32
        label_cond_in_channels = 3
        use_label_cond_conv = True
        pretrained_weights_path = '../runs/hnn_CT_conv_before_concat__DA_tumorW_0_25_3_2025_14:02:50/checkpoints/hnn_001000.pt'  # Specify the correct path
            
        model = get_model(in_channels=in_channels, 
                        label_cond_in_channels=label_cond_in_channels, 
                        use_label_cond_conv=use_label_cond_conv,
                        pretrained_weights_path=pretrained_weights_path)
        model.eval()
        model.cuda()

        dl, ds, data_set = get_loader(image_key=image_key, 
                                    label_key=label_key, 
                                    clip_min=clip_min, 
                                    clip_max=clip_max, 
                                    image_size=image_size, 
                                    no_seg=no_seg, 
                                    full_background=full_background, 
                                    data_split_json=data_split_json, 
                                    base_dir=base_dir)
        
        scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]
        root_output_path = "../results/Synthetic_Datasets/Whole_scans/Tumour_generation/conv_before_concat"
        for sch in scheduler_list:
            os.makedirs(os.path.join(root_output_path, sch), exist_ok=True)
        
        num_inference_steps = 100
        
        
        run_inference_conv_before_concat(model=model,
                    scheduler_list=scheduler_list,
                    n=len(dl), 
                    num_inference_steps=num_inference_steps, 
                    clip_min=clip_min,
                    clip_max=clip_max, 
                    root_output_path=root_output_path,
                    dl=dl,
                    cycle_id=cycle_id)            
        
#### hnn_CT_concat_cond__DA_tumorW_0_28_11_2024_11:36:09 #### 
#* HU between -200 and 200. tumour weight 0. DA ROI. downsampled ROI and segmentation as condition.

def run_inference_concat_cond(model, scheduler_list, n, num_inference_steps, clip_min, clip_max, root_output_path, dl, cycle_id):
    model.cuda()
    for idx, batch  in enumerate(dl):
        print(f"Case number {idx}")
        if "scan_ct_meta_dict" in batch:
            case_path = batch['scan_ct_meta_dict']['filename_or_obj'][0]
            seg_path = batch['seg_meta_dict']['filename_or_obj'][0]
        elif "image_meta_dict" in batch:
            case_path = batch['image_meta_dict']['filename_or_obj'][0]
            seg_path = batch['seg_meta_dict']['filename_or_obj'][0]
        print(f"Loaded scan {case_path}")
        print(f"Loaded seg {seg_path}")

        if "empty" not in seg_path:
            case_name = case_path.split('/')[-1].split(".nii.gz")[0]
            for sch in scheduler_list:
                out_path = os.path.join(root_output_path, sch)
                scheduler = get_scheduler(sch, num_inference_steps)

                noise_start = torch.randn(1, 8, 128, 128, 128)  
                # Prepare the noisy image
                final_scan = noise_start.clone().detach()
                final_scan = final_scan.cuda()

                label_condition = batch["seg"].cuda()
                segmentation = label_condition[0][2]
                no_contrast_tensor = label_condition[0][0]
                contrast_tensor = label_condition[0][1]

                # create input model
                resize = Resize((128, 128, 128), size_mode='all', mode="nearest", align_corners=None, anti_aliasing=False, anti_aliasing_sigma=None, dtype=torch.float32, lazy=False)
                label_cond_down = resize(label_condition[0]).unsqueeze(0)
                input_model = torch.cat((final_scan, label_cond_down), dim=1)
                input_model = input_model.cuda()

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
                        input_model = torch.cat((final_scan, label_cond_down), dim=1)
                B, C, D, H, W = final_scan.size()
                final_scan = idwt(final_scan[:, 0, :, :, :].view(B, 1, H, W, D) * 3.,
                            final_scan[:, 1, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 2, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 3, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 4, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 5, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 6, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 7, :, :, :].view(B, 1, H, W, D))
                # Assuming final_image is a PyTorch tensor
                # Convert the final_image tensor to a NumPy array if it's a tensor
                final_image_np = final_scan.squeeze().cpu().numpy()  # Remove the channel dim and move to CPU

                if th.sum(contrast_tensor) != 0:
                    cube_coords = th.nonzero(contrast_tensor) 
                    min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                    max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                else:
                    cube_coords = th.nonzero(no_contrast_tensor) 
                    min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                    max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                
                sample_denorm = np.clip(final_image_np, a_min=-1, a_max=1) # remove very high and low values

                sample_denorm = rescale_array(
                                arr=sample_denorm, 
                                minv=int(clip_min), 
                                maxv=int(clip_max)
                                )
                # Cropping the output of the model considering the ROI
                x_min, y_min, z_min = min_coords
                x_max, y_max, z_max = max_coords
                sample_denorm_corrected = sample_denorm[x_min:x_max, y_min:y_max, z_min:z_max]
                segmentation = segmentation[x_min:x_max, y_min:y_max, z_min:z_max]

                affine, header = get_affine_and_header(case_path)

                sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1) 
                synth_ct_scan_output = os.path.join(out_path, f'{case_name}_CT_n{cycle_id}.nii.gz')
                nii_image = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header)
                nib.save(nii_image, synth_ct_scan_output)

                segmentation = np.flip(segmentation.cpu().numpy(), axis=1) 
                nii_image = nib.Nifti1Image(segmentation, affine=affine, header=header)
                seg_ct_scan_output = os.path.join(out_path, f'{case_name}_label_n{cycle_id}.nii.gz')
                nib.save(nii_image, seg_ct_scan_output)
        
        else:
            print(f"{case_path.split('/')[-1].split('.nii.gz')[0]} DONE")
        if idx+1 == n:
            break
            
def main_concat_cond(number_of_cylces): 
    for cycle_id in range(number_of_cylces):    
        # Fixed for CT
        image_key = "image"
        label_key = "seg"
        image_size = (256, 256, 256)
        data_split_json = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json"
        base_dir = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256"
        full_background = False
        no_seg = False

        # To change
        clip_min = -200
        clip_max = 200
        in_channels = 11
        label_cond_in_channels = 0
        use_label_cond_conv = False
        pretrained_weights_path = '../runs/hnn_CT_concat_cond__DA_tumorW_0_25_3_2025_14:02:26/checkpoints/hnn_001000.pt'
            
        model = get_model(in_channels=in_channels, 
                        label_cond_in_channels=label_cond_in_channels, 
                        use_label_cond_conv=use_label_cond_conv,
                        pretrained_weights_path=pretrained_weights_path)
        model.eval()
        model.cuda()

        dl, ds, data_set = get_loader(image_key=image_key, 
                                    label_key=label_key, 
                                    clip_min=clip_min, 
                                    clip_max=clip_max, 
                                    image_size=image_size, 
                                    no_seg=no_seg, 
                                    full_background=full_background, 
                                    data_split_json=data_split_json, 
                                    base_dir=base_dir)
        print("Loaded model and data loader")

        # Control inference parameters
        scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]
        root_output_path = "../results/Synthetic_Datasets/Whole_scans/Tumour_generation/concat_cond"
        for sch in scheduler_list:
            os.makedirs(os.path.join(root_output_path, sch), exist_ok=True)
        num_inference_steps = 100
        
        
        run_inference_concat_cond(model=model,
                    scheduler_list=scheduler_list,
                    n=len(dl), 
                    num_inference_steps=num_inference_steps, 
                    clip_min=clip_min,
                    clip_max=clip_max, 
                    root_output_path=root_output_path,
                    dl=dl,
                    cycle_id=cycle_id) 
        
#### hnn_CT_wavelet_cond__DA_tumorW_0_3_12_2024_15:36:29 ####
#* HU between -200 and 200. tumour weight 0. DA ROI. wavelet tranformed ROI and segmentation as condition.
def run_inference_wavelet_cond(model, scheduler_list, n, num_inference_steps, clip_min, clip_max, root_output_path, dl, cycle_id):
    model.cuda()
    for idx, batch  in enumerate(dl):
        print(f"Case number {idx}")
        if "scan_ct_meta_dict" in batch:
            case_path = batch['scan_ct_meta_dict']['filename_or_obj'][0]
            seg_path = batch['seg_meta_dict']['filename_or_obj'][0]
        elif "image_meta_dict" in batch:
            case_path = batch['image_meta_dict']['filename_or_obj'][0]
            seg_path = batch['seg_meta_dict']['filename_or_obj'][0]
        print(f"Loaded scan {case_path}")
        print(f"Loaded seg {seg_path}")

        if "empty" not in seg_path:
            case_name = case_path.split('/')[-1].split(".nii.gz")[0]

            for sch in scheduler_list:
                out_path = os.path.join(root_output_path, sch)
                scheduler = get_scheduler(sch, num_inference_steps)

                noise_start = torch.randn(1, 8, 128, 128, 128)  
                # Prepare the noisy image
                final_scan = noise_start.clone().detach()
                final_scan = final_scan.cuda()

                label_condition = batch["seg"].cuda()
                segmentation = label_condition[0][2]
                no_contrast_tensor = label_condition[0][0]
                contrast_tensor = label_condition[0][1]

                LLL = None
                # create input model
                for condition in label_condition[0]:
                    condition = condition.unsqueeze(0).unsqueeze(0)
                    if LLL==None:
                        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(condition)
                        cond_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
                    else:
                        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(condition)
                        cond_dwt = th.cat([cond_dwt, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
                input_model = torch.cat((final_scan, cond_dwt), dim=1)
                input_model = input_model.cuda()

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
                        input_model = torch.cat((final_scan, cond_dwt), dim=1)
                B, C, D, H, W = final_scan.size()
                final_scan = idwt(final_scan[:, 0, :, :, :].view(B, 1, H, W, D) * 3.,
                            final_scan[:, 1, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 2, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 3, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 4, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 5, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 6, :, :, :].view(B, 1, H, W, D),
                            final_scan[:, 7, :, :, :].view(B, 1, H, W, D))
                # Assuming final_image is a PyTorch tensor
                # Convert the final_image tensor to a NumPy array if it's a tensor
                final_image_np = final_scan.squeeze().cpu().numpy()  # Remove the channel dim and move to CPU

                if th.sum(contrast_tensor) != 0:
                    
                    cube_coords = th.nonzero(contrast_tensor) 
                    min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                    max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                else:
                    
                    cube_coords = th.nonzero(no_contrast_tensor) 
                    min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                    max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                
                
                sample_denorm = np.clip(final_image_np, a_min=-1, a_max=1) # remove very high and low values

                sample_denorm = rescale_array(
                                arr=sample_denorm, 
                                minv=int(clip_min), 
                                maxv=int(clip_max)
                                )
                # Cropping the output of the model considering the ROI
                x_min, y_min, z_min = min_coords
                x_max, y_max, z_max = max_coords
                sample_denorm_corrected = sample_denorm[x_min:x_max, y_min:y_max, z_min:z_max]
                segmentation = segmentation[x_min:x_max, y_min:y_max, z_min:z_max]

                affine, header = get_affine_and_header(case_path)

                sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1) 
                synth_ct_scan_output = os.path.join(out_path, f'{case_name}_CT_n{cycle_id}.nii.gz')
                nii_image = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header)
                nib.save(nii_image, synth_ct_scan_output)

                segmentation = np.flip(segmentation.cpu().numpy(), axis=1) 
                nii_image = nib.Nifti1Image(segmentation, affine=affine, header=header)
                seg_ct_scan_output = os.path.join(out_path, f'{case_name}_label_n{cycle_id}.nii.gz')
                nib.save(nii_image, seg_ct_scan_output)
        
        else:
            print(f"{case_path.split('/')[-1].split('.nii.gz')[0]} DONE")
        if idx+1 == n:
            break
                
def main_wavelet_cond(number_of_cylces):
    for cycle_id in range(number_of_cylces):          
        # Fixed for CT
        image_key = "image"
        label_key = "seg"
        image_size = (256, 256, 256)
        data_split_json = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json"
        base_dir = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256"
        full_background = False
        no_seg = False

        # To change
        clip_min = -200
        clip_max = 200
        in_channels = 32
        label_cond_in_channels = 0
        use_label_cond_conv = False
        pretrained_weights_path = '../runs/hnn_CT_wavelet_cond__DA_tumorW_0_25_3_2025_14:02:14/checkpoints/hnn_001000.pt'  # Specify the correct path
            
        model = get_model(in_channels=in_channels, 
                        label_cond_in_channels=label_cond_in_channels, 
                        use_label_cond_conv=use_label_cond_conv,
                        pretrained_weights_path=pretrained_weights_path)
        model.eval()
        model.cuda()

        dl, ds, data_set = get_loader(image_key=image_key, 
                                    label_key=label_key, 
                                    clip_min=clip_min, 
                                    clip_max=clip_max, 
                                    image_size=image_size, 
                                    no_seg=no_seg, 
                                    full_background=full_background, 
                                    data_split_json=data_split_json, 
                                    base_dir=base_dir)
        print("Loaded model and data loader")

        # Control inference parameters
        scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]
        root_output_path = "../results/Synthetic_Datasets/Whole_scans/Tumour_generation/wavelet_cond"
        for sch in scheduler_list:
            os.makedirs(os.path.join(root_output_path, sch), exist_ok=True)
        
        num_inference_steps = 100
        
        
        run_inference_wavelet_cond(model=model,
                    scheduler_list=scheduler_list,
                    n=len(dl), 
                    num_inference_steps=num_inference_steps, 
                    clip_min=clip_min,
                    clip_max=clip_max, 
                    root_output_path=root_output_path,
                    dl=dl,
                    cycle_id=cycle_id) 
    

#### hnn_CT_conv_before_concat__DA_tumorW_0_28_11_2024_11:19:14 ####
#* HU between -200 and 200. tumour weight 0. DA ROI. ROI and segmentation as condition, feeded first to a conv layer.
#main_conv_before_concat(number_of_cylces=1)

#### hnn_CT_concat_cond__DA_tumorW_0_28_11_2024_11:36:09 #### 
#* HU between -200 and 200. tumour weight 0. DA ROI. downsampled ROI and segmentation as condition.
#main_concat_cond(number_of_cylces=1)

#### hnn_CT_wavelet_cond__DA_tumorW_0_3_12_2024_15:36:29 ####
#* HU between -200 and 200. tumour weight 0. DA ROI. wavelet tranformed ROI and segmentation as condition.
#main_wavelet_cond(number_of_cylces=1)



# --------------------------------------------------------------------------- #
### Bone segmentation ###
#### hnn_CT_concat_cond__DA_tumorW_0_28_11_2024_14:15:39 ####
#* HU between -200 and 200. DA ROI. downsampled ROI as condition. 
def run_inference_concat_cond_bone_200(model, scheduler_list, n, num_inference_steps, clip_min, clip_max, root_output_path, dl, cycle_id):
    model.cuda()
    for idx, batch  in enumerate(dl):
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
            noise_start = torch.randn(1, 8, 128, 128, 128)  
            # Prepare the noisy image
            final_scan = noise_start.clone().detach()
            final_scan = final_scan.cuda()

            label_condition = batch["seg"].cuda()
            segmentation = label_condition[0][2]
            no_contrast_tensor = label_condition[0][0]
            contrast_tensor = label_condition[0][1]

            label_condition =label_condition[:,0:2,:,:,:]
            print(f"label_condition: {label_condition.shape}")

            # create input model
            resize = Resize((128, 128, 128), size_mode='all', mode="nearest", align_corners=None, anti_aliasing=False, anti_aliasing_sigma=None, dtype=torch.float32, lazy=False)
            label_cond_down = resize(label_condition[0]).unsqueeze(0)
            input_model = torch.cat((final_scan, label_cond_down), dim=1)
            input_model = input_model.cuda()

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
                    input_model = torch.cat((final_scan, label_cond_down), dim=1)
            B, C, D, H, W = final_scan.size()
            final_scan = idwt(final_scan[:, 0, :, :, :].view(B, 1, H, W, D) * 3.,
                        final_scan[:, 1, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 2, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 3, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 4, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 5, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 6, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 7, :, :, :].view(B, 1, H, W, D))
            # Assuming final_image is a PyTorch tensor
            # Convert the final_image tensor to a NumPy array if it's a tensor
            final_image_np = final_scan.squeeze().cpu().numpy()  # Remove the channel dim and move to CPU

            if th.sum(contrast_tensor) != 0:
                cube_coords = th.nonzero(contrast_tensor) 
                min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                segmentation = contrast_tensor
            else:
                cube_coords = th.nonzero(no_contrast_tensor) 
                min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                segmentation = no_contrast_tensor
            
            sample_denorm = np.clip(final_image_np, a_min=-1, a_max=1) # remove very high and low values
            
            sample_denorm = rescale_array(
                            arr=sample_denorm, 
                            minv=int(clip_min), 
                            maxv=int(clip_max)
                            )
            # Cropping the output of the model considering the ROI
            x_min, y_min, z_min = min_coords
            x_max, y_max, z_max = max_coords
            sample_denorm_corrected = sample_denorm[x_min:x_max, y_min:y_max, z_min:z_max]
            segmentation = segmentation[x_min:x_max, y_min:y_max, z_min:z_max]

            affine, header = get_affine_and_header(case_path)

            sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1) 
            synth_ct_scan_output = os.path.join(out_path, f'{case_name}_CT_n{cycle_id}.nii.gz')
            nii_image = nib.Nifti1Image(sample_denorm_corrected, affine=affine, header=header)
            nib.save(nii_image, synth_ct_scan_output)

           
        if idx+1 == n:
            break

def main_concat_cond_bone_200(number_of_cylces):
    for cycle_id in range(number_of_cylces):           
        # Fixed for CT
        image_key = "image"
        label_key = "seg"
        image_size = (256, 256, 256)
        data_split_json = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json"
        base_dir = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256"
        full_background = False
        no_seg = False

        # To change
        clip_min = -200
        clip_max = 200
        in_channels = 10
        label_cond_in_channels = 0
        use_label_cond_conv = False
        pretrained_weights_path = '../runs/hnn_CT_concat_cond__DA_tumorW_0_25_3_2025_14:13:26/checkpoints/hnn_001000.pt'  # Specify the correct path
            
        model = get_model(in_channels=in_channels, 
                        label_cond_in_channels=label_cond_in_channels, 
                        use_label_cond_conv=use_label_cond_conv,
                        pretrained_weights_path=pretrained_weights_path)
        model.eval()
        model.cuda()

        dl, ds, data_set = get_loader(image_key=image_key, 
                                    label_key=label_key, 
                                    clip_min=clip_min, 
                                    clip_max=clip_max, 
                                    image_size=image_size, 
                                    no_seg=no_seg, 
                                    full_background=full_background, 
                                    data_split_json=data_split_json, 
                                    base_dir=base_dir)
        print("Loaded model and data loader")

        # Control inference parameters
        scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]
        root_output_path = "../results/Synthetic_Datasets/Whole_scans/Bone_segmentation/200"
        for sch in scheduler_list:
            os.makedirs(os.path.join(root_output_path, sch), exist_ok=True)
        num_inference_steps = 100
        
        run_inference_concat_cond_bone_200(model=model,
                    scheduler_list=scheduler_list,
                    n=len(dl), 
                    num_inference_steps=num_inference_steps, 
                    clip_min=clip_min,
                    clip_max=clip_max, 
                    root_output_path=root_output_path,
                    dl=dl,
                    cycle_id=cycle_id) 
    
#### hnn_CT_concat_cond__DA_tumorW_0_28_11_2024_14:18:20 ####
#* HU between -1000 and 1000. DA ROI. downsampled ROI as condition.
def run_inference_concat_cond_bone_1000(model, scheduler_list, n, num_inference_steps, clip_min, clip_max, root_output_path, dl, cycle_id):
    model.cuda()
    for idx, batch  in enumerate(dl):
        print(f"Case number {idx}")
        if "scan_ct_meta_dict" in batch:
            case_path = batch['scan_ct_meta_dict']['filename_or_obj'][0]
        elif "image_meta_dict" in batch:
            case_path = batch['image_meta_dict']['filename_or_obj'][0]
        print(f"Loaded {case_path}")
        # Set case id
        case_name = case_path.split('/')[-1].split(".nii.gz")[0]
        noise_start = torch.randn(1, 8, 128, 128, 128)  
        for sch in scheduler_list:
            out_path = os.path.join(root_output_path, sch)
            scheduler = get_scheduler(sch, num_inference_steps)
            # Prepare the noisy image
            final_scan = noise_start.clone().detach()
            final_scan = final_scan.cuda()

            label_condition = batch["seg"].cuda()
            segmentation = label_condition[0][2]
            no_contrast_tensor = label_condition[0][0]
            contrast_tensor = label_condition[0][1]

            label_condition =label_condition[:,0:2,:,:,:]
            print(f"label_condition: {label_condition.shape}")


            # create input model
            resize = Resize((128, 128, 128), size_mode='all', mode="nearest", align_corners=None, anti_aliasing=False, anti_aliasing_sigma=None, dtype=torch.float32, lazy=False)
            label_cond_down = resize(label_condition[0]).unsqueeze(0)
            input_model = torch.cat((final_scan, label_cond_down), dim=1)
            input_model = input_model.cuda()

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
                    input_model = torch.cat((final_scan, label_cond_down), dim=1)
            B, C, D, H, W = final_scan.size()
            final_scan = idwt(final_scan[:, 0, :, :, :].view(B, 1, H, W, D) * 3.,
                        final_scan[:, 1, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 2, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 3, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 4, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 5, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 6, :, :, :].view(B, 1, H, W, D),
                        final_scan[:, 7, :, :, :].view(B, 1, H, W, D))
            # Assuming final_image is a PyTorch tensor
            # Convert the final_image tensor to a NumPy array if it's a tensor
            final_image_np = final_scan.squeeze().cpu().numpy()  # Remove the channel dim and move to CPU

            if th.sum(contrast_tensor) != 0:
                cube_coords = th.nonzero(contrast_tensor) 
                min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                segmentation = contrast_tensor
            else:
                cube_coords = th.nonzero(no_contrast_tensor) 
                min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
                segmentation = no_contrast_tensor
            
            sample_denorm = np.clip(final_image_np, a_min=-1, a_max=1) # remove very high and low values

            sample_denorm = rescale_array(
                            arr=sample_denorm, 
                            minv=int(clip_min), 
                            maxv=int(clip_max)
                            )
            # Cropping the output of the model considering the ROI
            x_min, y_min, z_min = min_coords
            x_max, y_max, z_max = max_coords
            sample_denorm_corrected = sample_denorm[x_min:x_max, y_min:y_max, z_min:z_max]
            segmentation = segmentation[x_min:x_max, y_min:y_max, z_min:z_max]
            
            affine, header = get_affine_and_header(case_path)

            sample_denorm_corrected = np.flip(sample_denorm_corrected, axis=1) 
            synth_ct_scan_output = os.path.join(out_path, f'{case_name}_CT_n{cycle_id}.nii.gz')
            nii_image = nib.Nifti1Image(sample_denorm_corrected,  affine=affine, header=header)  
            nib.save(nii_image, synth_ct_scan_output)

        if idx+1 == n:
            break
            
def main_concat_cond_bone_1000(number_of_cylces): 
    for cycle_id in range(number_of_cylces):               
        # Fixed for CT
        image_key = "image"
        label_key = "seg"
        image_size = (256, 256, 256)
        data_split_json = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json"
        base_dir = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256"
        full_background = False
        no_seg = False

        # To change
        clip_min = -1000
        clip_max = 1000
        in_channels = 10
        label_cond_in_channels = 0
        use_label_cond_conv = False
        pretrained_weights_path = '../runs/hnn_CT_concat_cond__DA_tumorW_0_25_3_2025_14:13:49/checkpoints/hnn_001000.pt'  # Specify the correct path
            
        model = get_model(in_channels=in_channels, 
                        label_cond_in_channels=label_cond_in_channels, 
                        use_label_cond_conv=use_label_cond_conv,
                        pretrained_weights_path=pretrained_weights_path)
        model.eval()
        model.cuda()

        dl, ds, data_set = get_loader(image_key=image_key, 
                                    label_key=label_key, 
                                    clip_min=clip_min, 
                                    clip_max=clip_max, 
                                    image_size=image_size, 
                                    no_seg=no_seg, 
                                    full_background=full_background, 
                                    data_split_json=data_split_json, 
                                    base_dir=base_dir)
        print("Loaded model and data loader")

        root_output_path = "../results/Synthetic_Datasets/Whole_scans/Bone_segmentation/1000"

        # Control inference parameters
        scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]
        for sch in scheduler_list:
            os.makedirs(os.path.join(root_output_path, sch), exist_ok=True)

        
        num_inference_steps = 100
        
        
        run_inference_concat_cond_bone_1000(model=model,
                    scheduler_list=scheduler_list,
                    n=len(dl), 
                    num_inference_steps=num_inference_steps, 
                    clip_min=clip_min,
                    clip_max=clip_max, 
                    root_output_path=root_output_path,
                    dl=dl,
                    cycle_id=cycle_id) 
        
#### hnn_CT_concat_cond__DA_tumorW_0_28_11_2024_14:15:39 ####
#* HU between -200 and 200. DA ROI. downsampled ROI as condition.
#main_concat_cond_bone_200(number_of_cylces=1)

#### hnn_CT_concat_cond__DA_tumorW_0_28_11_2024_14:18:20 ####
#* HU between -1000 and 1000. DA ROI. downsampled ROI as condition.
#main_concat_cond_bone_1000(number_of_cylces=1)