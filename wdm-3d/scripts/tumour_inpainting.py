import sys
sys.path.append(".")
sys.path.append("..")
import os
import argparse
import pathlib
from monai.transforms import Compose, LoadImage, CropForeground, EnsureChannelFirst, ResizeWithPadOrCrop, ScaleIntensityRange
from guided_diffusion.c_unet import UNetModel
import torch
import torch as th
from diffusers import DPMSolverMultistepScheduler
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from scipy.ndimage import binary_dilation  
from scipy.ndimage import center_of_mass  
import json 
idwt = IDWT_3D("haar")

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

def get_model(model_path):
    model = UNetModel(
        image_size=128,
        in_channels=4,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=tuple([]),
        dropout=0.0,
        channel_mult=[1, 2, 2, 4, 4],
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
        label_cond_in_channels=0,
        use_label_cond_conv=False,
    )

    state_dict = torch.load(model_path, map_location=torch.device('cuda:0'))  # Load to CPU, or adjust for GPU if needed
    model.load_state_dict(state_dict)
    print("Loaded Model")
    return model

def get_scheduler(args):
    # Get the scheduler
    if args.scheduler=="DPM++_2M":
        use_karras_sigmas = False
        algorithm_type = "dpmsolver++"
    elif args.scheduler=="DPM++_2M_Karras":
        use_karras_sigmas = True
        algorithm_type = "dpmsolver++"
    elif args.scheduler=="DPM++_2M_SDE":
        use_karras_sigmas = False
        algorithm_type = "sde-dpmsolver++"
    elif args.scheduler=="DPM++_2M_SDE_Karras":
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
    scheduler.set_timesteps(num_inference_steps=args.sampling_steps)
    return scheduler

def get_input_noise(args):
    if args.use_wavelet:
        print(f"THIS IS WRONG!!")
        wavelet_coefficients = int(len(args.modality.split("_"))*8)
        img = th.randn(args.batch_size,         # Batch size
                    wavelet_coefficients,    # 8 wavelet coefficients
                    args.image_size//2,      # Half spatial resolution (D)
                    args.image_size//2,      # Half spatial resolution (H)
                    args.image_size//2,      # Half spatial resolution (W)
                    ).cuda()
    else:
        img = th.randn(args.batch_size,         # Batch size
                        1,    
                        args.image_size,      # Half spatial resolution (D)
                        args.image_size,      # Half spatial resolution (H)
                        args.image_size,      # Half spatial resolution (W)
                        ).cuda()
    return img
 
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
    x = torch.linspace(-3, 3, blur_factor)
    kernel_1d = torch.exp(-0.5 * x**2 / sigma**2)
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

def main(args):
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Get pre-trained model
    model = get_model(args.model_path)
    model.eval()
    model.cuda()

    with open('../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json', 'r') as file:
        training_cases = json.load(file)
        training_cases = training_cases['training']
    datal = training_cases 
    
    
    # datal = [] # TODO for real cases
    for case in training_cases:
        training_cases
        if "empty" not in case['seg']:
            datal.append(case)
            
    print(f"Number of cases for inference: {len(datal)}")
    # Get list of segmentations
    args.seg_path = "../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/seg"
    with open(args.json_file, 'r') as file:
        data_train = json.load(file)
        data_train = data_train['training']
    allowed_segs = []
    for case in data_train:
        seg_file = case['seg'].replace('seg/','')
        allowed_segs.append(seg_file)

    segs_list = []
    for case in os.listdir(args.seg_path):
        if "empty" not in case:
            if case in allowed_segs:
                segs_list.append(case)
    print(f"segs_list cases: {len(segs_list)}")

    schedulers= ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]


    for ind, batch in enumerate(datal):
        case_name = f"{batch['image'].split('/')[-1].split('.nii.gz')[0]}"
        
        for scheduler_name in schedulers:
            data_dir_here = os.path.join(args.data_dir, scheduler_name) # Synthetic data TODO
            #data_dir_here = args.data_dir # Original data TODO
            output_dir_here = os.path.join(args.output_dir, scheduler_name)
            os.makedirs(output_dir_here, exist_ok=True)

            case_path = os.path.join(data_dir_here, f"{case_name}_CT_n0.nii.gz") # TODO
            #case_path = os.path.join(args.data_dir, batch['image']) # Original data # Original data TODO
            #seg_path = os.path.join(args.data_dir, batch['seg']) # Original data TODO: to get the correspondent segmentation

            print(f"Loaded {case_path}")
            # Get mask file (place to insert the tumour)
            mask_file_path = os.path.join(f"../results/Synthetic_Datasets/Whole_scans/Bone_segmentation/Mask_for_tumour_inpaint", args.output_dir.split('/')[-1], scheduler_name, f"{case_name}_CT_n0_tumour_place.nii.gz")
            
            # Get contrast 
            contrast_value = batch["contrast"]

            # Defining the img (noise) input to the model
            noise_start = get_input_noise(args)
            final_scan = noise_start.clone().detach()
            final_scan = final_scan.cuda()
            

            if case_path.endswith("_CT_n0.nii.gz"):
                print("Random tumour")
                # Prepar data for inference
                seg_name = random.choice(segs_list) 
                seg_path = os.path.join(args.seg_path, seg_name) 
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

            
            label_condition = th.cat((no_contrast_tensor, contrast_tensor, label_crop_pad), dim=1) 
            
            print(f"Case_name: {case_name.split('_')[1]}")
            

            if args.use_dilation or args.use_mask_blur:
                # Perform binary dilation
                label_crop_pad_dillated = label_crop_pad.cpu().detach().clone().numpy().squeeze()
                structuring_element = np.ones((3, 3, 3), dtype=str)
                dilated_mask = binary_dilation(label_crop_pad_dillated, structure=structuring_element, iterations=5)
                dilated_mask = torch.from_numpy(dilated_mask).float()
                dilated_mask = dilated_mask.unsqueeze(dim=0).unsqueeze(dim=0).cuda()
                if args.use_mask_blur:
                    # Blur mask -> better to blend in the tumour :D
                    blurred_mask = blur_mask_3d(dilated_mask, label_crop_pad, blur_factor=25, blur_type=args.use_mask_blur)
                else:
                    blurred_mask = dilated_mask
            else:
                blurred_mask = label_crop_pad.detach().clone()

            # Assuming final_image is a PyTorch tensor
            # Convert the final_image tensor to a NumPy array if it's a tensor
            input_model = torch.cat((noise_start, label_condition), dim=1).cuda()
            healthy_ct_scan_original = healthy_ct_scan.detach().clone()
            
            scheduler = get_scheduler(args)

            # Start the reverse process (denoising from noise)
            for j, timestep in enumerate(tqdm(scheduler.timesteps)):
                # Get the current timestep's noise
                t = torch.tensor([timestep] * noise_start.shape[0]).cuda()
                t.cuda()
                # Perform one step of denoising
                with torch.no_grad():
                    model_kwargs = {}
                    model_prediction = model(input_model, timesteps=t, label_condition=label_condition, **model_kwargs)
                    
                    # Update the noisy_latents (reverse the noise process)
                    final_scan = scheduler.step(model_output=model_prediction, timestep=t, sample=final_scan)
                    final_scan = final_scan['prev_sample']

                    # Add noise to the healthy image
                    noise = torch.randn(1, 1, 128, 128, 128).cuda()                
                    healthy_ct_scan_noisy = scheduler.add_noise(
                        original_samples=healthy_ct_scan_original, 
                        noise=noise, 
                        timesteps=t)
                    # Replace the healthy ROI with the model prediction
                    noisy_image = healthy_ct_scan_noisy * (1 - blurred_mask) + final_scan * blurred_mask
                    input_model = torch.cat((noisy_image, label_condition), dim=1).cuda()
            
            # The at timestep 0 the final_scan is the scan without noise and with the tumour
            # Although, let's make a last replacment of the ROI just to avoid any irregularities in the surrounding tissue
            final_image = healthy_ct_scan_original * (1 - blurred_mask) + final_scan * blurred_mask

            # Create the full resolution scan with synthetic tumour
            sample_denorm = np.clip(final_image.detach().cpu().numpy()[0][ :, :, :], a_min=-1, a_max=1)
            sample_denorm = rescale_array(
                        arr=sample_denorm, 
                        minv=(healthy_ct_scan_origin_intensities.cpu().numpy().min()), 
                        maxv=(healthy_ct_scan_origin_intensities.cpu().numpy().max())
                        )
            if th.sum(contrast_tensor) != 0:
                ending_name = "contrast"
                cube_coords = th.nonzero(contrast_tensor[0, 0]) 
                min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
            else:
                ending_name = "no_contrast"
                cube_coords = th.nonzero(no_contrast_tensor[0, 0]) 
                min_coords = cube_coords.min(dim=0)[0]  # Minimum x, y, z coordinates
                max_coords = cube_coords.max(dim=0)[0]  # Maximum x, y, z coordinates
            
            x_min, y_min, z_min = min_coords
            x_max, y_max, z_max = max_coords
            sample_denorm_corrected = sample_denorm[x_min:x_max, y_min:y_max, z_min:z_max]

            # Get affine and header
            affine, header = get_affine_and_header(file_path=case_path)
            
            # Save the full resolution scan with the synthetic tumour
            new_ct_scan_full_res_np =  np.copy(healthy_ct_scan_full_res.detach().cpu().numpy())
            new_ct_scan_full_res_np = new_ct_scan_full_res_np[0][0]
            new_ct_scan_full_res_np[sagittal[0]:sagittal[1], coronal[0]:coronal[1], axial[0]:axial[1]] = sample_denorm
            output_name = os.path.join(output_dir_here, f"{case_name}_CT_n0.nii.gz")
            img = nib.Nifti1Image(new_ct_scan_full_res_np[64:-64, 64:-64, 64:-64,], affine=affine, header=header) 
            nib.save(img=img, filename=output_name)
            # 
            new_label_full_res_np = np.zeros_like(new_ct_scan_full_res_np)
            new_label_full_res_np[sagittal[0]:sagittal[1], coronal[0]:coronal[1], axial[0]:axial[1]] = label_crop_pad[0][0].float().detach().cpu().numpy()
            output_name = os.path.join(output_dir_here, f"{case_name}_label_n0.nii.gz")
            img = nib.Nifti1Image(new_label_full_res_np[64:-64, 64:-64, 64:-64,], affine=affine, header=header) 
            nib.save(img=img, filename=output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tumour Inpainting")
    # Add expected arguments
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of input data")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for processing")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading")
    parser.add_argument('--renormalize', type=str, default=None, help="Renormalization option")
    parser.add_argument('--empty_seg', type=str, default=True, help="Whether to use empty segmentation")
    parser.add_argument('--full_background', type=str, default=True, help="Whether to use full background")
    parser.add_argument('--clip_min', type=int, default=-200, help="Minimum clipping value")
    parser.add_argument('--clip_max', type=int, default=200, help="Maximum clipping value")
    parser.add_argument('--use_wavelet', type=str, default=False, help="Whether to use wavelet transformation")
    parser.add_argument('--image_size', type=int, default=128, help="Size of the image")
    parser.add_argument('--use_dilation', type=str, default=True, help="Whether to use dilation")
    parser.add_argument('--use_data_augmentation', type=str, default=False, help="Whether to use data augmentation")
    parser.add_argument('--modality', type=str, default="CT", help="Modality used (e.g., CT)")
    parser.add_argument('--train_mode', type=str, default="tumour_inpainting", help="Mode of training")
    parser.add_argument('--mode', type=str, default="c_sample", help="Mode of operation")
    parser.add_argument('--dataset', type=str, default="hnn_tumour_inpainting", help="Dataset name")
    parser.add_argument('--sampling_steps', type=int, default=100, help="Number of sampling steps")
    parser.add_argument('--scheduler', type=str, default="DPM++_2M", help="Scheduler type")
    parser.add_argument('--use_mask_blur', type=str, default=True, help="Whether to use mask blur")
    parser.add_argument('--json_file', type=str, default=True, help="What json file with data splits to use")


    # Parse the arguments passed to the script
    args = parser.parse_args()
    print(f"args.use_wavelet: {args.use_wavelet}")
    # Handling arguments to ensure they are strean
    args.empty_seg = True if (args.empty_seg=="True" or args.empty_seg==True) else False
    args.full_background = True if (args.full_background=="True" or args.full_background==True) else False
    args.use_wavelet = True if (args.use_wavelet=="True" or args.use_wavelet==True) else False
    print(f"args.use_wavelet: {args.use_wavelet}")
    args.use_dilation = True if (args.use_dilation=="True" or args.use_dilation==True) else False
    args.use_data_augmentation = True if (args.use_data_augmentation=="True" or args.use_data_augmentation==True) else False
    args.use_mask_blur = True if (args.use_mask_blur=="True" or args.use_mask_blur==True) else False
    
    main(args)