import sys
import os
import warnings
from os import listdir, makedirs, environ
from os.path import join, exists, dirname, basename

import json
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import argparse
import random

import torch
import torch.nn as nn
from torch import optim
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

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
    ScaleIntensityd
    )

from monai.data import load_decathlon_datalist, DataLoader, CacheDataset, Dataset
from monai.transforms.transform import MapTransform
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from collections.abc import Callable, Hashable, Mapping

#####################
## Utils 
####################
def maybe_make_dir(directory):
    if not exists(directory):
        # If it doesn't exist, create the directory
        makedirs(directory)

def create_loss_jpeg(txt_loss_path, iter_per_epoch, val_mode=False, val_size=None):
    """
    Takes a txt file and crates the plot in jpeg format

    ARGUMENTS:
        txt_loss_path: Path to the txt file with the losses

    RETURN:
        jpeg file with the plot of losses
    """
    save_path = dirname(txt_loss_path)
    file_name = basename(txt_loss_path).split(".")[0]
    # Read the content of the text file
    with open(txt_loss_path, "r") as file:
        content = file.read()
    # Convert the content into a list of integers
    number_list = [float(number) for number in content.split()]

    plt.figure(figsize=(20, 12)) 
    plt.plot(number_list)
    #Save the plot as a JPEG file
    plt.savefig(join(save_path, f"{file_name}_total.jpg"))
    plt.close()
    
    plt.figure(figsize=(20, 12))
    # Create a plot with the mean per epoch
    loss_per_epoch = []
    for epochs in range(0, (len(number_list)//iter_per_epoch)):
        somador = 0
        for i in range(iter_per_epoch*epochs, iter_per_epoch*(epochs+1)):
            somador += number_list[i]
        media = somador/iter_per_epoch
        loss_per_epoch.append(media)

    plt.plot(loss_per_epoch, label='Train')
    
    loss_per_epoch = []
    if val_mode:
        val_path = join(CHECKPOINT_DIR, "loss_lists", f"{val_mode}.txt")
        with open(val_path, "r") as file:
            val_content = file.read()
        val_values = [float(number) for number in val_content.split()]

        for epochs in range(0, (len(val_values)//val_size)): 
            somador = 0
            for i in range(val_size*epochs, val_size*(epochs+1)):
                somador += val_values[i]
            media = somador/val_size
            loss_per_epoch.append(media)
        plt.plot(loss_per_epoch, label='Val')
        # Adding a legend
        plt.legend()
        
        
    #Save the plot as a JPEG file
    plt.savefig(join(save_path, f"{file_name}_epoch.jpg"))
    plt.close()

def create_loss_jpeg_simple(txt_loss_path, iter_per_epoch, val_mode=False, val_size=None):
    """
    Takes a txt file and crates the plot in jpeg format

    ARGUMENTS:
        txt_loss_path: Path to the txt file with the losses

    RETURN:
        jpeg file with the plot of losses
    """
    save_path = dirname(txt_loss_path)
    file_name = basename(txt_loss_path).split(".")[0]
    # Read the content of the text file
    with open(txt_loss_path, "r") as file:
        content = file.read()
    # Convert the content into a list of integers
    number_list = [float(number) for number in content.split()]

    plt.figure(figsize=(20, 12)) 
    plt.plot(number_list)
    #Save the plot as a JPEG file
    plt.savefig(join(save_path, f"{file_name}_total.jpg"))
    plt.close()

def save_loss_file(loss_name, loss_list, reset_loss_file, val_mode=False, val_size=None, jpeg_simple=False):
    """
    Save the loss into a txt file. If the file exists, it appends the content

    ARGUMENTS:
        loss_name: Name of the loss to save the txt file
        loss_list: List of losses to write into the txt file
        reset_loss_file: To create a new txt file

    RETURN:
        A {loss_name}.txt with the loss list
        A jpeg file with the plot os the losses
    """
    
    file_path = join(CHECKPOINT_DIR, "loss_lists", f"{loss_name}.txt")
    # Check if the file exists
    if not exists(file_path) or reset_loss_file:
        # If the file doesn't exist, create it
        with open(file_path, 'w') as new_file:
            new_file.write("")

   # Append the new list to the existing content
    with open(file_path, 'a') as file:
        file.write(" " + " ".join(loss_list))

    if jpeg_simple:
        create_loss_jpeg_simple(txt_loss_path=file_path, iter_per_epoch=len(loss_list), val_mode=val_mode, val_size=val_size)
    else:
        create_loss_jpeg(txt_loss_path=file_path, iter_per_epoch=len(loss_list), val_mode=val_mode, val_size=val_size)

def seed_everything(seed=1):
    environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#####################
## DATA LOADER 
####################
class ConvertToMultiChannelBasedOnBratsClasses2023_here(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 1) | (img == 3), (img == 1) | (img == 2) | (img == 3), img == 3]
        # merge labels 1 (tumor non-enh) and 3 (tumor enh) and 2 (large edema) to WT
        # label 3 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class ConvertToMultiChannelBasedOnBratsClasses2023d_here(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsClasses2023_here.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClasses2023_here()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def get_data_split(all_cases):
    # Set a seed for reproducibility (optional)
    np.random.seed(321)
    # Shuffle the indices of the cases
    shuffled_indices = np.random.permutation(len(all_cases))

    # Define the split ratio
    split_ratio = 0.95 
    split_index = int(len(all_cases) * split_ratio)

    # Split the indices
    train_indices = shuffled_indices[:split_index]
    val_indices = shuffled_indices[split_index:]
    
    # Create training and validation sets
    train_set = [all_cases[i] for i in train_indices]
    val_set = [all_cases[i] for i in val_indices]
    
    return train_set, val_set


def generate_detection_train_transform(
    image_key,
    label_key,
    image_size,
    AMP=True,
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
    compute_dtype = torch.float32
    train_transforms = Compose(
        [
            LoadImaged(keys=[image_key, label_key], meta_key_postfix="meta_dict", image_only=False),
            EnsureChannelFirstd(keys=[image_key, label_key]),
            EnsureTyped(keys=[image_key, label_key], dtype=torch.float32),
            Orientationd(keys=[image_key, label_key], axcodes="RAS"),
            ResizeWithPadOrCropd(
                    keys=[image_key, label_key],
                    spatial_size=image_size,
                    mode="constant",
                    value=0
                ),
            ScaleIntensityd(keys=[image_key], minv=-1, maxv=1),
            ConvertToMultiChannelBasedOnBratsClasses2023d_here(
                keys=[label_key],
            ),
            EnsureTyped(keys=[image_key, label_key], dtype=compute_dtype)
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=[image_key, label_key], meta_key_postfix="meta_dict", image_only=False),
            EnsureChannelFirstd(keys=[image_key, label_key]),
            EnsureTyped(keys=[image_key, label_key], dtype=torch.float32),
            Orientationd(keys=[image_key, label_key], axcodes="RAS"),
            ResizeWithPadOrCropd(
                    keys=[image_key, label_key],
                    spatial_size=image_size,
                    mode="constant",
                    value=-1
                ),
            ScaleIntensityd(keys=[image_key], minv=-1, maxv=1),
            ConvertToMultiChannelBasedOnBratsClasses2023d_here(
                keys=[label_key],
            ),
            EnsureTyped(keys=[image_key, label_key], dtype=compute_dtype)
        ]
    )
    return train_transforms, val_transforms

def get_loader():
    """
    ARGS:
        image_size: final image size for resizing 
        batch_size: Batch size
        
    RETURN:
        train_loader: data loader
        train_data: dict of the data loaded 
    """

    # Get train transforms
    train_transforms, val_transforms = generate_detection_train_transform(
            image_key = "t1c",
            label_key = "seg",
            image_size = IMAGE_SIZE,
        )

    # Get training data dict 
    all_data = load_decathlon_datalist(
            DATA_LIST_FILE_PATH,
            is_segmentation=True,
            data_list_key=DATA_LIST_KEY,
            base_dir=DATA_DIR,
        )

    train_set, val_set = get_data_split(all_cases=all_data[:]) 
    print(f"Training cases: {len(train_set)}")
    print(f"Validation cases: {len(val_set)}")
    
    # Creating traing dataset
    train_ds = CacheDataset( 
        data=train_set[:], 
        transform=train_transforms,
        cache_rate=1,
        copy_cache=False,
        progress=True,
        num_workers=NUM_WORKERS,
    )

    val_ds = CacheDataset(
        data=val_set[:],
        transform=val_transforms,
        cache_rate=1,
        copy_cache=False,
        progress=True,
        num_workers=NUM_WORKERS,
    )
    
    #train_ds = Dataset(train_data, transform=train_transforms)
    
    # Creating data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        #collate_fn=no_collation,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        #collate_fn=no_collation,
    )

    print(f"Batch size: {BATCH_SIZE}")
    return train_loader, val_loader, train_ds, val_ds

def get_nets_n_loader():
    """
    Creating all the objects needed for training
    
    ARGUMENTS:

    RETURN:
        gen: Generator
        disc: Discriminator
        opt_gen: Optimizer of the generator
        opt_critic: Optimizer of the discriminator
        loader: data loader
        dataset: dataset dict
    """
    if TEST:
        assert RESUME!=None, "Choose the checkpoint number in RESUME"
        print(f"RESUMING WEIGHTS FROM EPOCH {RESUME}")
        gen = Generator(in_channels=DIM, latent_dim=NOISE_DIM, IN_CHANNEL_G=IN_CHANNEL_G, OUT_CHANNEL_G=OUT_CHANNEL_G, z_dim=NOISE_DIM, w_dim=NOISE_DIM, skip_latent=SKIP_LATENT, tahn_act=TAHN_ACT)
        gen.to(DEVICE_G)
        opt_gen = optim.AdamW(gen.parameters(), lr=LR_G, betas=(0.5, 0.999))
        
        gen_weight_path = join(CHECKPOINT_DIR, "weights", f"{RESUME}_gen.pth")
        checkpoint = torch.load(gen_weight_path)
        # Load the model's state dictionary
        gen.load_state_dict(checkpoint['model_state_dict'])
        opt_gen.load_state_dict(checkpoint['optimizer_state_dict'])
        begin_epoch = checkpoint['epoch']
        model_name = checkpoint['name']
        gen.eval()
        
        loader, dataset = get_loader()
        return gen, opt_gen, loader, dataset, begin_epoch, model_name
    
    else:
        gen = Generator(in_channels=DIM, latent_dim=NOISE_DIM, IN_CHANNEL_G=IN_CHANNEL_G, OUT_CHANNEL_G=OUT_CHANNEL_G, z_dim=NOISE_DIM, w_dim=NOISE_DIM, skip_latent=SKIP_LATENT, tahn_act=TAHN_ACT)
        critic = Critic(in_channels=DIM, img_channels=IN_CHANNEL_D)

        gen = gen.to(DEVICE_G)
        critic = critic.to(DEVICE_D)
        gen.train()
        critic.train()
        opt_gen = optim.AdamW(gen.parameters(), lr=LR_G, betas=(0.5, 0.999))
        opt_critic = optim.AdamW(critic.parameters(), lr=LR_D, betas=(0.5, 0.999))

        begin_epoch = 0

        if RESUME!=None:
            """
            if UNET == "Unet_FC":
                gen_weight_path = join(CHECKPOINT_DIR, "weights", f"{RESUME}_gen.pth")
                checkpoint = torch.load(gen_weight_path, map_location=torch.device(DEVICE_G))
                try:
                    opt_gen.load_state_dict(checkpoint['optimizer_state_dict']) # This raises an error when the new architecture is distinct form the original one
                except:
                    warnings.warn("Optimizer State Dict not loaded!")

                begin_epoch = checkpoint['epoch']
                model_name = checkpoint['name']

                pretrain_dict = checkpoint["model_state_dict"]
                # Filter out incompatible layers
                target_dict = gen.state_dict()
                pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in target_dict}
                # Load the pre-trained weights into the target model
                del pretrain_dict["enc.block_1.conv1.conv.weight"]
                del pretrain_dict["dec.genBlock_7.conv1.conv.weight"]
                gen.load_state_dict(pretrain_dict, strict=False)
                gen.train()
            """

            #elif UNET == "Unet":
            print(f"RESUMING WEIGHTS FROM EPOCH {RESUME}")
            # Load the checkpoint Generator
            gen_weight_path = join(CHECKPOINT_DIR, "weights", f"{RESUME}_gen.pth")
            checkpoint = torch.load(gen_weight_path)
            # Load the model's state dictionary
            gen.load_state_dict(checkpoint['model_state_dict'])
            opt_gen.load_state_dict(checkpoint['optimizer_state_dict'])
            begin_epoch = checkpoint['epoch']
            model_name = checkpoint['name']
            gen.train()
            #End of elif

            # Load the checkpoint Critic
            gen_weight_path = join(CHECKPOINT_DIR, "weights", f"{RESUME}_critic.pth")
            checkpoint = torch.load(gen_weight_path)
            # Load the model's state dictionary
            critic.load_state_dict(checkpoint['model_state_dict'])
            opt_critic.load_state_dict(checkpoint['optimizer_state_dict'])
            begin_epoch = checkpoint['epoch']
            model_name = checkpoint['name']
            prob = checkpoint['prob']
            #prob = 0
            critic.train()

        train_loader, val_loader, train_ds, val_ds = get_loader()
        if RESUME:
            prob = prob
        else:
            prob = 0.0

        return critic, gen, opt_critic, opt_gen, train_loader, val_loader, train_ds, val_ds, begin_epoch, prob

## Update the prob if the val is negative and the train is positive!
def get_transforms(prob):
    if IN_CHANNEL_D != 1:
        keys_list = ["t1c", "seg"]
        mode_list = ['bilinear', 'nearest']
    elif IN_CHANNEL_D == 1:
        keys_list = ["t1c"]
        mode_list = ['bilinear']

    print(f"NEW PROB: {prob}")
    # Create a list of transforms
    transforms = [
        # Based on file:///Users/andreferreira/Downloads/s10462-023-10453-z.pdf and https://arxiv.org/pdf/2006.06676.pdf
        # rotate 6 degrees
        # translation (16,32,8)
        # scale_range (-0.1, 0.1) -> zoom!
        # shear_range (-0.1, 0.1)
        RandAffined(
            keys=keys_list, 
            prob=prob,
            rotate_range=((-np.pi/30,np.pi/30),(-np.pi/30,np.pi/30),(-np.pi/30,np.pi/30)), # 6 degrees
            translate_range=(16,16,16), 
            scale_range=((-0.2,0.2),(-0.2,0.2),(-0.2,0.2)),
            shear_range=((-0.2,0.2),(-0.2,0.2),(-0.2,0.2)),
            padding_mode="reflection",
            mode=mode_list,
            ),
        # flipping
        RandFlipd(keys=keys_list, prob=prob, spatial_axis=0), #sagittal
        RandFlipd(keys=keys_list, prob=prob, spatial_axis=1), #coronal
        RandFlipd(keys=keys_list, prob=prob, spatial_axis=2), # axial
        # Rotate 90 degrees
        RandRotate90d(keys=keys_list, prob=prob, max_k=3),
        #### Intensity ####
        # Contrast (Creates instability and the Generator creates Nan for some reason)
        #RandAdjustContrastd(keys=[keys_list[0]], prob=prob, gamma=(0.5, 2.0)),
        #Blur (I did not like the blur) NOT DO!!!!
        #RandGaussianSharpend(keys=["in_0", "in_1"], prob=1, ),
        #Noise gaussian
        #RandGaussianNoised(keys=[keys_list[0]], prob=prob, mean=0, std=0.5),
        # Salt and peper
        # NOT IMPLEMENTED
        # Random crop
        # NOT IMPLEMENTED
    ]
    
    # Combine the transforms into a single transform
    combined_transform = Compose(transforms)
    return combined_transform
    
def do_transforms(data, combined_transform):
    if IN_CHANNEL_D != 1:
        ct_scan, seg = data[0][0].unsqueeze(0), data[0][1:]
        in_to_transform = {
            "t1c": ct_scan,
            "seg": seg,
        }
    elif IN_CHANNEL_D == 1:
        ct_scan = data[0]
        in_to_transform = {
            "t1c": ct_scan, 
            }
    output_tensor = combined_transform(in_to_transform)
  
    if IN_CHANNEL_D != 1:
        out= torch.cat([output_tensor["t1c"], output_tensor["seg"]], dim=0).unsqueeze(0)
    elif IN_CHANNEL_D == 1:
        out = output_tensor["t1c"].unsqueeze(0)

    return out 

#####################
## TRAINING Utils 
####################
def gradient_penalty(critic, real, fake, combined_transform):
    real = real.to(DEVICE_D)
    fake = fake.to(DEVICE_D)
    BATCH_SIZE, C, H, W, D = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1, 1)).repeat(1, C, H, W, D).to(DEVICE_D)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    #mixed_scores = do_critic_infer(critic=critic, data=interpolated_images, combined_transform=combined_transform)
    trans_data = do_transforms(data=interpolated_images, combined_transform=combined_transform)
    mixed_scores = critic(trans_data)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=trans_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def do_gen_infer(gen, data):
    fake_image = gen(data)
    return fake_image

def do_critic_infer(critic, data, combined_transform):
    trans_data = do_transforms(data, combined_transform)
    critic_value = critic(trans_data)
    return critic_value

def get_critic_loss(gen, critic, seg, ct_scan, combined_transform):
    
        ### Generate a synthetic scan
        fake_image = do_gen_infer(gen=gen, data=seg.to(DEVICE_G))
            
        # Prepare fake and real input for the critic, with seg
        in_fake_critic = torch.cat([fake_image.to(DEVICE_D), seg.to(DEVICE_D)], dim=1)
        in_real_critic = torch.cat([ct_scan.to(DEVICE_D), seg.to(DEVICE_D)], dim=1)
        
        ### Compute fake loss
        critic_fake = do_critic_infer(critic=critic, data=in_fake_critic.detach().to(DEVICE_D), combined_transform=combined_transform)
        
        ### Compute real loss
        critic_real = do_critic_infer(critic=critic, data=in_real_critic.to(DEVICE_D), combined_transform=combined_transform)
        
        ### Compute gradient penalty 
        gp = gradient_penalty(critic=critic, real=in_real_critic, fake=in_fake_critic, combined_transform=combined_transform)

        # Total critic loss
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake)) 
            + W_GP * gp
            + (0.001 * torch.mean(critic_real ** 2))
        ) * W_ADV_D 
        return loss_critic, critic_fake, critic_real, gp, fake_image
    
def get_gen_loss(gen, critic, seg, ct_scan, fake_image, combined_transform, warmup):
    
        in_fake_gen = torch.cat((fake_image.to(DEVICE_D), seg.to(DEVICE_D)), dim=1)
        
        ### Critic judgment
        if not warmup:
            critic_gen_fake = do_critic_infer(critic=critic, data=in_fake_gen, combined_transform=combined_transform)
            critic_gen_fake_loss = -torch.mean(critic_gen_fake).to(DEVICE_G)
        else:
            critic_gen_fake_loss = torch.tensor([0])
            critic_gen_fake_loss = critic_gen_fake_loss.to(DEVICE_G)

        ### Voxel-wise loss 
        ct_scan = ct_scan.to(DEVICE_G)
        fake_image = fake_image.to(DEVICE_G)
        seg = seg.to(DEVICE_G)
        voxel_wise_all_loss = MAE_LOSS(ct_scan, fake_image)
        #### Overall tumour
        #### Just tumour
        if torch.sum(seg[0][1]) != 0:
            square = (ct_scan[0][0]*seg[0][1] - fake_image[0][0]*seg[0][1])**2
            voxel_wise_tumour_loss = torch.sum(square)/torch.sum(seg[0][1])
        else:
            voxel_wise_tumour_loss = torch.tensor([0.0], device=DEVICE_G)
            print(f"SEG IS ZERO!")

        ### Total loss and backpropagation
        loss_gen = critic_gen_fake_loss*W_ADV_G + voxel_wise_all_loss*W_PWA + voxel_wise_tumour_loss*W_PWT
        return loss_gen, critic_gen_fake_loss, voxel_wise_all_loss, voxel_wise_tumour_loss

def get_val_critic(critic, ct_scan, seg):
    in_real_critic = torch.cat([ct_scan, seg], dim=1)
    critic_real = critic(in_real_critic)
    return critic_real  
    
def do_critic_step(gen, critic, seg, ct_scan, opt_critic, combined_transform, warmup):
    opt_critic.zero_grad(set_to_none=True)
    
    loss_critic, critic_fake, critic_real, gp, fake_image = get_critic_loss(gen=gen, critic=critic, seg=seg, ct_scan=ct_scan, combined_transform=combined_transform)
    if not warmup:
        # Call the GradScaler.scale() method to scale the loss on device
        loss_critic.backward()
        # Perform the backward pass on each device
        opt_critic.step()
        # Update the GradScaler for the next iteration on each device
        #scaler_d.update()
        opt_critic.zero_grad(set_to_none=True)
    return loss_critic, critic_fake, critic_real, gp, fake_image

def do_gen_step(gen, critic, seg, ct_scan, opt_gen, fake_image, combined_transform, warmup):
    opt_gen.zero_grad(set_to_none=True)
   
    loss_gen, critic_gen_fake_loss, voxel_wise_all_loss, voxel_wise_tumour_loss = get_gen_loss(gen=gen, critic=critic, seg=seg, ct_scan=ct_scan, fake_image=fake_image, combined_transform=combined_transform, warmup=warmup)
    # Call the GradScaler.scale() method to scale the loss on GPU 0
    loss_gen.backward()
    # Perform the backward pass 
    opt_gen.step()
    opt_gen.zero_grad(set_to_none=True)
    return loss_gen, critic_gen_fake_loss, voxel_wise_all_loss, voxel_wise_tumour_loss
    
def one_iter(batch, critic, gen, opt_critic, opt_gen, list_of_losses, combined_transform, prob, epoch, warmup):
    ## Getting the real data
    
    #ct_scan, seg, ct_scan_2, seg_2 = batch["ct"][0].unsqueeze(0), batch["seg"][0].unsqueeze(0),  batch["ct"][1].unsqueeze(0),  batch["seg"][1].unsqueeze(0)
    ct_scan, seg = batch["t1c"], batch["seg"]

    ###########################################################
    ################### CRITIC STEP ###########################
    ###########################################################
    loss_critic, critic_fake, critic_real, gp, fake_image = do_critic_step(gen=gen, critic=critic, seg=seg, ct_scan=ct_scan, opt_critic=opt_critic, combined_transform=combined_transform, warmup=warmup)
    
    ###########################################################
    ###################### GENERATOR STEP #####################
    ###########################################################
    loss_gen, critic_gen_fake_loss, voxel_wise_all_loss, voxel_wise_tumour_loss = do_gen_step(gen=gen, critic=critic, seg=seg, ct_scan=ct_scan, opt_gen=opt_gen, fake_image=fake_image, combined_transform=combined_transform, warmup=warmup)
    

    list_of_losses[0].append(str(loss_critic.item()))
    list_of_losses[1].append(str(critic_real.item()))
    list_of_losses[2].append(str(critic_fake.item()))
    list_of_losses[3].append(str(gp.item()))

    list_of_losses[4].append(str(loss_gen.item()))
    list_of_losses[5].append(str(critic_gen_fake_loss.item()))
    list_of_losses[6].append(str(voxel_wise_all_loss.item()))
    list_of_losses[7].append(str(voxel_wise_tumour_loss.item()))

    loop_train.set_postfix(
        critic = loss_critic.item(),
        real_c = critic_real.item(),
        fake_c = critic_fake.item(),
        gp = gp.item(),
        gen = loss_gen.item(),
        fake_g = critic_gen_fake_loss.item(),
        voxel_wise_A = voxel_wise_all_loss.item(),
        voxel_wise_T = voxel_wise_tumour_loss.item()
    )

    if batch_idx % 1000 == 0:
        with torch.no_grad():
            
            fake_scan_control = do_gen_infer(gen=gen, data=seg.to(DEVICE_G))
        np_fake_scan_control = np.squeeze((fake_scan_control[0]).data.cpu().numpy()).astype(np.float32)
        affine = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],  
                   [0, 0, 0, 1]])
        nifti_fake_scan_control = nib.Nifti1Image(np_fake_scan_control, affine=affine)
        #plotting.plot_img(nifti_img, title="F", cut_coords=None, annotate=False, draw_cross=False, black_bg=True)
        nib.save(nifti_fake_scan_control, join(CHECKPOINT_DIR, f"nifti/epoch_{epoch}_iter_{batch_idx}_fake.nii.gz"))

        np_fake_scan_control = np.squeeze((ct_scan[0]).data.cpu().numpy()).astype(np.float32)
        affine = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],  
                   [0, 0, 0, 1]])
        nifti_fake_scan_control = nib.Nifti1Image(np_fake_scan_control, affine=affine)
        #plotting.plot_img(nifti_img, title="F", cut_coords=None, annotate=False, draw_cross=False, black_bg=True)
        nib.save(nifti_fake_scan_control, join(CHECKPOINT_DIR, f"nifti/epoch_{epoch}_iter_{batch_idx}_real.nii.gz"))
        
        for cond_label_idx, cond_label_in in enumerate(seg[0]):
            np_fake_scan_control = np.squeeze((cond_label_in).data.cpu().numpy()).astype(np.float32)
            nifti_fake_scan_control = nib.Nifti1Image(np_fake_scan_control, affine=affine)
            #plotting.plot_img(nifti_img, title="F", cut_coords=None, annotate=False, draw_cross=False, black_bg=True)
            nib.save(nifti_fake_scan_control, join(CHECKPOINT_DIR, f"nifti/epoch_{epoch}_iter_{batch_idx}_label_{cond_label_idx}.nii.gz"))
            

    return list_of_losses, combined_transform, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--RESUME", default=None, required=False, help="If resume != None, i.e., resume == an integer, networks weights will be loaded.")
    parser.add_argument("--TEST", type=str, default="False", required=False, help="For testing purposes. In case of True, the discrimnator is not loaded and the generator is loaded in eval(). Resume must be != None")
    parser.add_argument("--NUM_WORKERS", type=int, default=8, required=False, help="Number of workers for the data loader")
    parser.add_argument("--BATCH_SIZE", type=int, default=1, required=False, help="Batch size. Bigger than 1 will consume a lot of memory.")
    parser.add_argument("--NOISE_DIM", type=int, default=512, required=False, help="Size of the latent space")
    parser.add_argument("--IN_CHANNEL_G", type=int, default=2, required=False, help="Number of input channels for the Generator")
    parser.add_argument("--OUT_CHANNEL_G", type=int, default=1, required=False, help="Number of output channels for the Generator")
    parser.add_argument("--IN_CHANNEL_D", type=int, default=2, required=False, help="Number of input channels for the Discriminator. 2 in case of conditional discriminator (recommended)")
    parser.add_argument("--DIM", type=int, default=1024, required=False, help="Number max of dimentions.This can be reduced in case of lack of memory")
    parser.add_argument("--LR_G", type=float, default=0.0001, required=False, help="Learning rate of the Generator")
    parser.add_argument("--LR_D", type=float, default=0.0001, required=False, help="Learning rate of the Discriminator")
    parser.add_argument("--W_ADV_D", type=int, default=1, required=False, help="Weight of the Adversarial component of the loss function of the Discriminator")
    parser.add_argument("--W_ADV_G", type=int, default=1, required=False, help="Weight of the Adversarial component of the loss function of the Generator")
    parser.add_argument("--W_GP", type=int, default=10, required=False, help="Weight of the Gradient Penalty component of the loss function of the Discriminator")
    parser.add_argument("--W_PWA", type=int, default=100, required=False, help="Weight of the whole image pixel wise component of the loss function of the Generator")
    parser.add_argument("--W_PWT", type=int, default=10, required=False, help="Weight of the tumour pixel wise component of the loss function of the Generator")
    parser.add_argument("--TOTAL_EPOCHS", type=int, default=200, required=False, help="Total number of epochs")
    parser.add_argument("--WARMUP_EPOCHS", type=int, default=0, required=False, help="Total number of epochs to warmup the generator, by using only the pixel wise metrics")
    parser.add_argument("--EXP_NAME", type=str, default="TRASH", required=True, help="Name of the experiment")
    parser.add_argument("--DATASET", type=str, default="training", required=False, help="Name of the JSON file containing the data paths")
    parser.add_argument("--LR_DECAY", type=str, default="False", required=False, help="If want to use leraning rate decay of AdamW. NOT IMPLEMENTED!")
    parser.add_argument("--DA", type=str, default="False", required=False, help="If want to use conventional data augmentation")
    parser.add_argument("--GEN_ITER", type=int, default=1, required=False, help="How many times train the generator per iteration")
    parser.add_argument("--DISC_ITER", type=int, default=1, required=False, help="How many times train the discriminator per iteration")
    parser.add_argument("--SKIP_LATENT", type=str, default="True", required=False, help="If want to skip connect of the latent tensor")
    parser.add_argument("--TAHN_ACT", type=str, default="False", required=False, help="If want to use tanh acttivation in the generator")
    parser.add_argument("--UNET", type=str, default="Unet", required=False, help="If want to use Unet like Generator. You can choose Unet and Unet_FC")
   
    
    args = parser.parse_args()

    if args.SKIP_LATENT=="True":
        print("USING SKIP LATENT TO THE ADAIN")
        SKIP_LATENT = True
    else:
        SKIP_LATENT = False

    if args.TAHN_ACT=="True":
        print("USING TANH ACTIVATION IN THE GENERATOR")
        TAHN_ACT = True
    else:
        TAHN_ACT = False

    UNET = args.UNET

    ### IF RESUMING OR TESTING ###
    RESUME = args.RESUME
    if args.RESUME!=None:
        INIT_EPOCH = int(RESUME)
    else:
        INIT_EPOCH = 0

    if args.TEST == "True":
        TEST = True
    else:
        TEST = False

    ### NETWORK SPECIFICATIONS ###
    BATCH_SIZE = args.BATCH_SIZE
    NOISE_DIM = args.NOISE_DIM
    IN_CHANNEL_G = args.IN_CHANNEL_G
    OUT_CHANNEL_G = args.OUT_CHANNEL_G
    IN_CHANNEL_D = args.IN_CHANNEL_D
    DIM = args.DIM

    # Network specifications 
    
    ### DATA LOADER ###
    NUM_WORKERS = args.NUM_WORKERS
    if args.DA == "True":
        DA = True
    else:
        DA = False
    DATASET = args.DATASET

    ### LOSS FUNCTION ###
    W_ADV_D = args.W_ADV_D
    W_ADV_G = args.W_ADV_G
    W_GP = args.W_GP
    W_PWA = args.W_PWA
    W_PWT = args.W_PWT

    ### LEARNING SCHEDULE ###
    LR_G = args.LR_G
    LR_D = args.LR_D
    if args.LR_DECAY=="True":
        LR_DECAY = True
    else:
        LR_DECAY = False

    ### TRAINING ###
    TOTAL_EPOCHS = args.TOTAL_EPOCHS
    WARMUP_EPOCHS = int(args.WARMUP_EPOCHS)
    EXP_NAME = args.EXP_NAME
    GEN_ITER = args.GEN_ITER
    DISC_ITER = args.DISC_ITER

    ##############################
    ###### FIXED ARGUMENTS #######
    ##############################

    # Defining working directory
    HOME_DIR = "/projects"
    WORK_DIR = "/projects"
    HPCWORK = "/projects"

    # Device
    DEVICE_G = "cuda:0"
    DEVICE_D = "cuda:1"
    # Last imports
    sys.path.insert(1, join(HOME_DIR, "aritifcial-head-and-neck-cts/GANs/src"))
    if UNET=="Unet_FC":
        print("USING THE UNET Fully Connected LIKE GENERATOR")
        from network.cWGAN_Style_Unet_256_FC import Generator, Critic
    elif UNET=="Unet":
        print("USING THE UNET LIKE GENERATOR")
        from network.cWGAN_Style_Unet_256 import Generator, Critic
    else:
        print("WELCOME TO THE ERROR ZONE")

    # json file
    DATA_LIST_FILE_PATH = join(WORK_DIR, "aritifcial-head-and-neck-cts/GANs/data/BraTS2023_GLI_data_split.json") # Path where to save the json file 

    # Checkpoints dir
    CHECKPOINT_DIR = join(HPCWORK, "aritifcial-head-and-neck-cts", "GANs", "checkpoint/style_256", EXP_NAME)

    # Dataset folder and vars
    DATA_DIR = "../brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" 

    DATA_LIST_KEY = "training"

    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True

    IMAGE_SIZE = (256, 256, 256)

    #MSE_LOSS = torch.nn.MSELoss() # Pixel-wise loss metric
    MAE_LOSS = torch.nn.L1Loss() # Pixel-wise loss metric

    # Set if new txt are created for the loss values
    if args.RESUME==None:
        reset_loss_file = True
    else:
        reset_loss_file = False


    LIST_LOSS_NAMES = ["disc_real_val_L", "l_critic_L", "l_real_c_L", "l_fake_c_L", "l_gp_L", "l_gen_L", "l_fake_g_L", "l_voxel_wise_A_L", "l_voxel_wise_T_L"]

    # Functions call 
    CREAT_DIRS = True
    CALL_SEED_EVERYTHING = True
    DEBUG = False

    # List of directories to create
    DIRS_TO_CREATE = [CHECKPOINT_DIR, join(CHECKPOINT_DIR, "loss_lists"), join(CHECKPOINT_DIR, "weights"), join(CHECKPOINT_DIR, "nifti")]

    if CREAT_DIRS:
        for directory in DIRS_TO_CREATE:
            maybe_make_dir(directory)

    if CALL_SEED_EVERYTHING:
        seed_everything()

    # Loading all the objects needed for training 
    critic, gen, opt_critic, opt_gen, train_loader, val_loader, train_ds, val_ds, begin_epoch, prob = get_nets_n_loader()
    val_size = len(val_ds)
    print(f"DATA AUGMENTATION: {DA}")
    
    
    combined_transform = get_transforms(prob=prob)


    for epoch in range(INIT_EPOCH, TOTAL_EPOCHS):
        loop_train = tqdm(train_loader, leave=True)
        l_critic_L, l_real_c_L, l_fake_c_L, l_gp_L, l_gen_L, l_fake_g_L, l_voxel_wise_A_L, l_voxel_wise_T_L = [], [], [], [], [], [], [], []
        list_of_losses = [l_critic_L, l_real_c_L, l_fake_c_L, l_gp_L, l_gen_L, l_fake_g_L, l_voxel_wise_A_L, l_voxel_wise_T_L]
        print("#########################")
        print(f"####### EPOCH {epoch} #######")
        print("#########################")
        if WARMUP_EPOCHS > epoch:
            print(f"Warming UP")
            warmup = True
        else:
            warmup = False

        for batch_idx, batch in enumerate(loop_train):
            list_of_losses, combined_transform, prob = one_iter(batch=batch, critic=critic, gen=gen, opt_critic=opt_critic, opt_gen=opt_gen, list_of_losses=list_of_losses, combined_transform=combined_transform, prob=prob, epoch=epoch, warmup=warmup)
        
        loop_val = tqdm(val_loader, leave=True)
        critic.eval()
        gen.eval()
        disc_real_val_L = []
        with torch.no_grad():
            print(f"##################")
            print("VALIDATION STARTED")
            for batch_idx, batch in enumerate(loop_val):
                ct_scan, seg  = batch["t1c"].to(DEVICE_D), batch["seg"].to(DEVICE_D)
                critic_real = get_val_critic(critic, ct_scan, seg)
                disc_real_val_L.append(str(critic_real.item()))

        if DA:
            ##### Deciding if increase the prob of data augmentation #####
            disc_real_val_L_float = [float(value) for value in disc_real_val_L]
            disc_real_val_L_sign = sum(disc_real_val_L_float) / len(disc_real_val_L_float)
            l_real_c_L_float = [float(value) for value in list_of_losses[1]]
            l_real_c_L_sign = sum(l_real_c_L_float) / len(l_real_c_L_float)
            # In case of the train real cases have positive output from the discriminator, but the validation is negative --> overfitting, so, increase prob
            
            if l_real_c_L_sign > 0 and disc_real_val_L_sign < 0:
                if prob < 1.0:
                    prob += 0.05
            else:
                if prob > 0.0:
                    prob -= 0.05
            if prob > 1.0:
                prob = 1.0
            elif prob < 0.0:
                prob = 0.0
        else:
            prob = 0.0
        combined_transform = get_transforms(prob=prob)

        list_of_losses.insert(0, disc_real_val_L)
        critic.train()
        gen.train()
        print("VALIDATION FINISHED")
        print(f"###################")

        assert len(LIST_LOSS_NAMES)==len(list_of_losses), "The list_of_losses does not correspond to the LIST_LOSS_NAMES"
        for loss_name, loss_list in zip(LIST_LOSS_NAMES, list_of_losses):
            if loss_name=="l_real_c_L":
                print(f"loss_name: {loss_name}")
                save_loss_file(loss_name=loss_name, loss_list=loss_list, reset_loss_file=reset_loss_file, val_mode="disc_real_val_L", val_size=val_size)
            else:
                print(f"loss_name: {loss_name}")
                save_loss_file(loss_name=loss_name, loss_list=loss_list, reset_loss_file=reset_loss_file, val_size=val_size)

        save_loss_file(loss_name="prob", loss_list=[str(prob)], reset_loss_file=reset_loss_file, val_size=val_size, jpeg_simple=True)
                
        reset_loss_file = False
        ## Save the models
        checkpoint_gen = {
        'name': "Generator",
        'epoch': epoch,
        'model_state_dict': gen.state_dict(),
        'optimizer_state_dict': opt_gen.state_dict(),
        }
        torch.save(checkpoint_gen, join(CHECKPOINT_DIR, "weights", f'{epoch}_gen.pth'))
        
        checkpoint_critic = {
        'name': "Critic",
        'epoch': epoch,
        'model_state_dict': critic.state_dict(),
        'optimizer_state_dict': opt_critic.state_dict(),
        'prob':prob,
        }
        torch.save(checkpoint_critic, join(CHECKPOINT_DIR, "weights", f'{epoch}_critic.pth'))
        print(f"MODELs SAVED")