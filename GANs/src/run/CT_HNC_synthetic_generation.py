EXPERIMENT_NAME = "W_PWA1000__W_PWT100__Unet_FC_min200_200"
UNET = "Unet_FC" # Unet or Unet_FC
RESUME = 999
DEVICE = "cuda:0"
HOME_DIR = "/projects"
WORK_DIR = "/projects"
IN_TYPE = "Contrast"

from os.path import join
# Fix variablles #
DATA_LIST_KEY = "training"
#DATA_LIST_KEY = "test"
DATA_LIST_FILE_PATH = join(WORK_DIR, "aritifcial-head-and-neck-cts/GANs/data/training.json") # Path where to save the json file 
DATA_DIR = join(WORK_DIR, "HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256")

CHECKPOINT_PATH = f"../checkpoint/style_256/{EXPERIMENT_NAME}"
IMAGE_SIZE = (256, 256, 256)
DIM = 1024
NOISE_DIM = 512
IN_CHANNEL_G = 3 # This should be changed in case of IN_TYPE = "Noise" or IN_TYPE = "Blank"
OUT_CHANNEL_G = 1
SKIP_LATENT = False
TAHN_ACT = False

### Saving the new cases
SAVE_DIR = "../nnUNet_seg/nnUNet_raw/Dataset983_synthCT_HNC/"


import sys
import torch
import numpy as np
import nibabel as nib
from nilearn import plotting
from scipy.ndimage import label
from tqdm import tqdm
from monai.data import load_decathlon_datalist, DataLoader, CacheDataset, Dataset
from monai.transforms import (
    Compose, 
    LoadImaged,
    EnsureChannelFirstd, 
    EnsureTyped,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    ScaleIntensityRanged
)

sys.path.insert(1, join(HOME_DIR, "aritifcial-head-and-neck-cts/GANs/src"))

# Load the GAN
if UNET=="Unet_FC":
    print("USING THE UNET Fully Connected LIKE GENERATOR")
    from network.cWGAN_Style_Unet_256_FC import Generator, Critic
elif UNET=="Unet":
    print("USING THE UNET LIKE GENERATOR")
    from network.cWGAN_Style_Unet_256 import Generator, Critic
else:
    print("WELCOME TO THE ERROR ZONE")

# Load the segmentation transformer
if IN_TYPE == "Noise":
    print("Using ConvertToMultiChannel_BackandForeground_blankd and adding noise")
    from utils.data_loader_utils import ConvertToMultiChannel_BackandForeground_blankd
    ConvertToMultiChannel_BackandForeground = ConvertToMultiChannel_BackandForeground_blankd
elif IN_TYPE == "Contrast":
    print("Using ConvertToMultiChannel_BackandForeground_Contrastd")
    from utils.data_loader_utils import ConvertToMultiChannel_BackandForeground_Contrastd
    ConvertToMultiChannel_BackandForeground = ConvertToMultiChannel_BackandForeground_Contrastd
elif  IN_TYPE == "Blank":
    print("Using ConvertToMultiChannel_BackandForeground_blankd")
    from utils.data_loader_utils import ConvertToMultiChannel_BackandForeground_blankd
    ConvertToMultiChannel_BackandForeground = ConvertToMultiChannel_BackandForeground_blankd

from os.path import join, exists, dirname, basename
from os import listdir, makedirs, environ
def maybe_make_dir(directory):
    if not exists(directory):
        # If it doesn't exist, create the directory
        makedirs(directory)

def get_gen(checkpoint_path, RESUME):
    print(f"Loading from: {checkpoint_path}")
    gen = Generator(in_channels=DIM, latent_dim=NOISE_DIM, IN_CHANNEL_G=IN_CHANNEL_G, OUT_CHANNEL_G=OUT_CHANNEL_G, z_dim=NOISE_DIM, w_dim=NOISE_DIM, skip_latent=SKIP_LATENT, tahn_act=TAHN_ACT)
    gen.to(DEVICE)
    gen_weight_path = join(checkpoint_path, "weights", f"{RESUME}_gen.pth")
    checkpoint = torch.load(gen_weight_path, map_location=torch.device(DEVICE))
    # Load the model's state dictionary
    gen.load_state_dict(checkpoint['model_state_dict'])
    gen.eval()
    return gen

def generate_detection_train_transform(
    image_key,
    label_key,
    image_size,
    ConvertToMultiChannel_BackandForeground, 
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
    transforms = Compose(
            [
                LoadImaged(keys=[image_key, label_key], meta_key_postfix="meta_dict", image_only=False),
                EnsureChannelFirstd(keys=[image_key, label_key]),
                EnsureTyped(keys=[image_key, label_key], dtype=torch.float32),
                Orientationd(keys=[image_key, label_key], axcodes="RAS"),
                ScaleIntensityRanged(keys=[image_key], a_min=-200, a_max=200.0, b_min=-1.0, b_max=1.0, clip=True),
                ResizeWithPadOrCropd(
                    keys=[image_key, label_key],
                    spatial_size=image_size,
                    mode="constant",
                    value=-1
                ),
                ConvertToMultiChannel_BackandForeground(
                    keys=[label_key],
                ),
                EnsureTyped(keys=[image_key, label_key], dtype=torch.float32)
            ]
        )

    return transforms

def get_loader(IMAGE_SIZE, DATA_LIST_KEY, DATA_DIR):
    """
    ARGS:
        image_size: final image size for resizing 
        batch_size: Batch size
        
    RETURN:
        train_loader: data loader
        train_data: dict of the data loaded 
    """

    # Get train transforms
    transforms = generate_detection_train_transform(
            image_key = "image",
            label_key = "seg",
            image_size = IMAGE_SIZE,
            ConvertToMultiChannel_BackandForeground = ConvertToMultiChannel_BackandForeground
        )

    # Get training data dict 
    data_set = load_decathlon_datalist(
            DATA_LIST_FILE_PATH,
            is_segmentation=True,
            data_list_key=DATA_LIST_KEY,
            base_dir=DATA_DIR,
        )
    print(data_set[0])
    ds = CacheDataset(
        data=data_set[:500],
        transform=transforms,
        cache_rate=1,
        copy_cache=False,
        progress=True,
        num_workers=4,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        #collate_fn=no_collation,
    )

    return loader, ds

def do_gen_infer(gen, data):
    fake_image = gen(data)
    return fake_image

def get_affine_header(file_path):
    nii_img = nib.load(file_path)
    affine_matrix = nii_img.affine
    header_info = nii_img.header
    return affine_matrix, header_info

def save_nifti(data, reality, affine=None, header_info=None,  save=None):
    if affine is None:
        affine = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],  # Assuming 3 for the spacing along the third axis
                   [0, 0, 0, 1]])
    try:
        np_fake = np.squeeze((data).data.cpu().numpy()).astype(np.float32)
    except:
        #print("Not torch!")
        np_fake = data
    nifti_fake = nib.Nifti1Image(np_fake, affine=affine, header=header_info)
    #plotting.plot_img(nifti_fake, title=reality, cut_coords=None, annotate=False, draw_cross=False, black_bg=True)
    if save!=None:
        nib.save(nifti_fake, save)

def save_nifti_with_metadata(exist_nii_path, new_numpy_array, save_path):
    existing_nii_file = nib.load(exist_nii_path)

    metadata = existing_nii_file.header
    affine = existing_nii_file.affine

    new_image = nib.Nifti1Image(new_numpy_array, affine, metadata)
    #new_image = nib.Nifti1Image(new_numpy_array, affine)
    nib.save(new_image, save_path)

def normalize_intensity(image, new_min, new_max):
    """
    Normalise the intensities into a new min and a new max 
    """
    # Assuming 'image' is a NumPy array with intensities in the range [-1, 1]
    clipped_image = np.clip(image, -1, 1)
    
    # Define the original range
    original_min = -1
    original_max = 1
    
    # Perform linear transformation to the new range
    normalized_image = (clipped_image - original_min) / (original_max - original_min) * (new_max - new_min) + new_min
    
    return normalized_image

def post_processing(fake_image, seg, ct_scan, new_min, new_max):
    """
    Performing post processing to the generated cases.
    Normalise intensity and crop.
    """
    fake_image_np = fake_image[0][0]#.cpu().numpy()
    ct_scan_np = ct_scan[0][0]#.cpu().numpy()
    
    if seg.shape[1]==3:
        # In case the segmentation is composed of 3 channels
        background_0 = seg[0][0].cpu().numpy()
        background_1 = seg[0][1].cpu().numpy()
        binary_seg = seg[0][2].cpu().numpy()
        if np.sum(background_0)!=0:
            #print("NO CONTRAST")
            binary_mask = background_0
        elif np.sum(background_1)!=0:
            #print("CONTRAST")
            binary_mask = background_1
        else:
            #print(f"All background is zero!")
            binary_mask = np.ones_like(binary_seg)
    elif seg.shape[1]==2:
        # In case the segmentation is composed of 2 channels  
        background = seg[0][0]#.cpu().numpy()
        binary_seg = seg[0][1]#.cpu().numpy()
        binary_mask = background

    # Normalise the scan intensities 
    fake_image_np_norm = normalize_intensity(image=fake_image_np, new_min=new_min, new_max=new_max)
    
    # Find connected components in the binary mask
    labeled_mask, num_components = label(binary_mask)
    # Assume that the region of interest is the largest connected component
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
    # Extract the bounding box of the largest connected component
    indices = np.where(labeled_mask == largest_component)
    min_x, max_x = np.min(indices[0]), np.max(indices[0])
    min_y, max_y = np.min(indices[1]), np.max(indices[1])
    min_z, max_z = np.min(indices[2]), np.max(indices[2])

    cropped_fake_scan = fake_image_np_norm[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
    cropped_seg = binary_seg[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
    cropped_ct_scan = ct_scan_np[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1] # TO remove

    # Flipping to have the same orientation as the original cases
    cropped_ct_scan = np.flip(cropped_ct_scan, axis=1)
    cropped_ct_scan = np.flip(cropped_ct_scan, axis=0)
    cropped_fake_scan = np.flip(cropped_fake_scan, axis=1)
    cropped_fake_scan = np.flip(cropped_fake_scan, axis=0)
    cropped_seg = np.flip(cropped_seg, axis=1)
    cropped_seg = np.flip(cropped_seg, axis=0)
    
    return cropped_ct_scan, cropped_fake_scan, cropped_seg

loader, ds = get_loader(IMAGE_SIZE=IMAGE_SIZE, DATA_LIST_KEY=DATA_LIST_KEY, DATA_DIR=DATA_DIR)

gen = get_gen(checkpoint_path=CHECKPOINT_PATH, RESUME=RESUME)

maybe_make_dir(directory=SAVE_DIR)
maybe_make_dir(directory=join(SAVE_DIR, "imagesTr"))
maybe_make_dir(directory=join(SAVE_DIR, "labelsTr"))

loop_train = tqdm(loader, leave=True)
for batch_idx, batch in enumerate(loop_train):
    with torch.no_grad():
        ct_scan, seg = batch["image"].to(DEVICE), batch["seg"].to(DEVICE)
        ct_path = batch["image_meta_dict"]["filename_or_obj"][0]
        seg_path = batch["seg_meta_dict"]["filename_or_obj"][0]
        ct_name = f"synt_{ct_path.split('/')[-1].split('.nii.gz')[0]}"
        #if "0a908279226c5229e7fe85b8894b62d5" in ct_name:
       
        if not torch.sum(seg[0][-1])==0:
            # Generating synthetic scan
            fake_image = do_gen_infer(gen=gen, data=seg)

            # Normalising synthetic scan intensity to the same values as the original case, 
            # and cropping to the same shape
            cropped_ct_scan, cropped_fake_scan, cropped_seg = post_processing(fake_image, seg, ct_scan, new_min=-200, new_max=200)

            # Saving synthetic scan
            save_path = join(SAVE_DIR, f"imagesTr/{ct_name}_0000.nii.gz")
            save_nifti_with_metadata(exist_nii_path=ct_path, new_numpy_array=cropped_fake_scan, save_path=save_path)
            
            # Saving segmentation
            save_path = join(SAVE_DIR, f"labelsTr/{ct_name}.nii.gz")
            
            save_nifti_with_metadata(exist_nii_path=seg_path, new_numpy_array=cropped_seg, save_path=save_path)
        else:
            pass
            #print(f"Ignored: {ct_name}")

        loop_train.set_postfix(
            Case = ct_name,
        )
        #print(ct_path)
        #print(seg_path)
      