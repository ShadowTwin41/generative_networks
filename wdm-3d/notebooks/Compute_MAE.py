## MAE - comparision between real case and generated case with the same condition (segmentation and/or ROI)
import os
import torch
import sys
import json
import warnings
from tqdm import tqdm
sys.path.append(".")
from test_utils import absolute_mean_error
from test_utils import get_data_loader

def get_CT_dataloader(clip_min, clip_max, fake_data_dir, pad_or_crop, load_only_fake):
    # Real data loader
    image_key = 'image'

    original_data_dir = "../../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256"
    # Load Training and Test data split
    with open("../../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json", 'r') as f:
        data_split = json.load(f)
        training = data_split['training']

    # Loading all real cases to memory
    real_image_files = []
    for file_name in training:
        if "Tumour_inpaint" in fake_data_dir or "Tumour_generation" in fake_data_dir:
            if 'empty' not in file_name['seg']:
                real_image_files.append(file_name['image'])
        else:
            real_image_files.append(file_name['image'])
    

    fake_image_files = []
    black_list = []
    for idx, img in enumerate(real_image_files):
        fake_name = img.split('/')[-1].replace('.nii.gz', '_CT_n0.nii.gz')
        fake_path = os.path.join(fake_data_dir, fake_name)
 
        if os.path.exists(fake_path):
            fake_image_files.append(fake_path)
        else:
            black_list.append(img.split('/')[-1])
            real_image_files.pop(idx)
    print(f"fake_image_files: {len(fake_image_files)}")
    print(f"real_image_files: {len(real_image_files)}")
    data_list = [{"image": img} for img in fake_image_files]
    print(f"fake data_list: {data_list[0]}")
    fake_data_loader = get_data_loader(image_key, clip_min, clip_max, data_list, pad_or_crop=pad_or_crop, two_loaders=False, cache_rate=[0])
    
    if not load_only_fake:       
        original_data_list = [{"image": os.path.join(original_data_dir, img)} for img in real_image_files]
        print(f"real original_data_list: {original_data_list[0]}") 
        real_data_loader = get_data_loader(image_key, clip_min, clip_max, original_data_list, pad_or_crop=pad_or_crop, two_loaders=False, cache_rate=[0]) # change 0 to 1
        return real_data_loader, fake_data_loader, black_list
    else:
        return fake_data_loader, black_list

def compute_MAE_folder_Bone(clip_min, clip_max, load_only_fake, pad_or_crop, fake_data_dir):
    # Check if the json file already exists
    json_file_path = os.path.join("./metrics/MAE", 
                                  fake_data_dir.split('Synthetic_Datasets/')[-1], 'MAE.json')
    
    if os.path.exists(json_file_path):
        print(f"Already DONE: {json_file_path}")
        return None
    else:
        real_data_loader, fake_data_loader, black_list = get_CT_dataloader(clip_min, clip_max, fake_data_dir, pad_or_crop=pad_or_crop, load_only_fake=load_only_fake)
        real_iter = iter(real_data_loader)
        mae_results = {}
        for fake_batch_idx, fake_batch in enumerate(fake_data_loader): # iterate over fake cases 
            real_batch = next(real_iter)
            case_name = fake_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].replace('_CT_n0.nii.gz', '.nii.gz')
            if case_name in black_list:
                print(f"case_name {case_name} in black list.")
                continue
            else:
                y_fake = fake_batch['image']
                y_real = real_batch['image']
                if case_name != real_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]:
                    warnings.warn(f"The cases don't match. Double check: {case_name} \n {real_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]}", SyntaxWarning)
                else:
                    try:
                        try:
                            mae_here = absolute_mean_error(y_real, y_fake)
                        except:
                            mae_here = absolute_mean_error(y_real[:,:,:-1,:-1,:-1], y_fake)
                    except:
                        try:
                            # Some cases don't have the same orientation. Therefore, they need to be flipped
                            mae_here = absolute_mean_error(y_real, torch.flip(y_fake.permute(0, 1, 2, 4, 3), dims=[4]))
                        except:
                            # Some cases don't have the same orientation. Therefore, they need to be flipped
                            mae_here = absolute_mean_error(y_real[:,:,:-1,:-1,:-1], torch.flip(y_fake.permute(0, 1, 2, 4, 3), dims=[4]))
                            
                    mae_results[case_name.replace('.nii.gz', '')] = float(mae_here)
                        
        print(f"Done: {mae_results}")
        
        
        os.makedirs(json_file_path.replace('MAE.json', ''), exist_ok=True)
        
        with open(json_file_path, "w") as json_file:
            json.dump(mae_results, json_file, indent=4)


def compute_MAE_folder_Tumour(clip_min, clip_max, load_only_fake, pad_or_crop, fake_data_dir):
    real_data_loader, fake_data_loader, black_list = get_CT_dataloader(clip_min, clip_max, fake_data_dir, pad_or_crop=pad_or_crop, load_only_fake=load_only_fake)
    real_iter = iter(real_data_loader)
    mae_results = {}
    for fake_batch_idx, fake_batch in enumerate(fake_data_loader): # iterate over fake cases 
        real_batch = next(real_iter)
        case_name = fake_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].replace('_CT_n0.nii.gz', '.nii.gz')
        if case_name in black_list:
            print(f"case_name {case_name} in black list.")
            continue
        else:
            y_fake = fake_batch['image']
            y_real = real_batch['image']
            while case_name != real_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]:
                real_batch = next(real_iter)
                warnings.warn(f"The cases don't match. Double check: {case_name} \n {real_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]}", SyntaxWarning)
            try:
                try:
                    mae_here = absolute_mean_error(y_real, y_fake)
                except:
                    mae_here = absolute_mean_error(y_real[:,:,:-1,:-1,:-1], y_fake)
            except:
                try:
                    # Some cases don't have the same orientation. Therefore, they need to be flipped
                    y_fake = y_fake.permute(0, 1, 2, 4, 3)
                    y_fake = torch.flip(y_fake, dims=[4])
                    mae_here = absolute_mean_error(y_real, y_fake)
                except:
                    # Some cases don't have the same orientation. Therefore, they need to be flipped
                    y_fake = y_fake.permute(0, 1, 2, 4, 3)
                    y_fake = torch.flip(y_fake, dims=[4])
                    mae_here = absolute_mean_error(y_real[:,:,:-1,:-1,:-1], y_fake)
            #pbar.set_postfix({"mae_here": float(mae_here)})
            mae_results[case_name.replace('.nii.gz', '')] = float(mae_here)
            
    print(f"Done: {mae_results}")
    json_file_path = os.path.join("./metrics/MAE", 
                                  fake_data_dir.split('Synthetic_Datasets/')[-1], 'MAE.json')
    
    os.makedirs(json_file_path.replace('MAE.json', ''), exist_ok=True)
    
    with open(json_file_path, "w") as json_file:
        json.dump(mae_results, json_file, indent=4)


# No change
load_only_fake = False
pad_or_crop=False
root_synthetic_data = "../results/Synthetic_Datasets/Whole_scans"

print("Starting...")
for experiment_fold in os.listdir(root_synthetic_data):
    experiment_path = os.path.join(root_synthetic_data, experiment_fold)
    if experiment_fold == "Bone_segmentation" and False:
        for hu_value in os.listdir(experiment_path):
            hu_value_path = os.path.join(experiment_path, hu_value)
            if hu_value=="200":
                clip_min = -200
                clip_max = 200
            elif hu_value=="1000":
                clip_min = -1000
                clip_max = 1000
            else:
                continue
            for scheduler in os.listdir(hu_value_path):
                fake_data_dir = os.path.join(hu_value_path, scheduler)
                print("#################")
                print(f"Doing {fake_data_dir}")
                compute_MAE_folder_Bone(clip_min, clip_max, load_only_fake, pad_or_crop, fake_data_dir)
    elif experiment_fold == "Tumour_generation" and False: # This is already done 
        clip_min = -200
        clip_max = 200
        for concat_method in os.listdir(experiment_path):
            concat_method_path = os.path.join(experiment_path, concat_method)
            for scheduler in os.listdir(concat_method_path):
                fake_data_dir = os.path.join(concat_method_path, scheduler)
                print("#################")
                print(f"Doing {fake_data_dir}")
                compute_MAE_folder_Tumour(clip_min, clip_max, load_only_fake, pad_or_crop, fake_data_dir)
    elif experiment_fold == "Tumour_inpaint"  and False:
        for mask_type in os.listdir(experiment_path):
            mask_type_path = os.path.join(experiment_path, mask_type)
            for hu_value in os.listdir(mask_type_path):
                hu_value_path = os.path.join(mask_type_path, hu_value)
                if hu_value=="200":
                    clip_min = -200
                    clip_max = 200
                elif hu_value=="1000":
                    clip_min = -1000
                    clip_max = 1000
                else:
                    continue
                for scheduler in os.listdir(hu_value_path):
                    fake_data_dir = os.path.join(hu_value_path, scheduler)
                    print("#################")
                    print(f"Doing {fake_data_dir}")
                    compute_MAE_folder_Bone(clip_min, clip_max, load_only_fake, pad_or_crop, fake_data_dir)


def get_CT_dataloader_GANs(clip_min, clip_max, fake_data_dir, pad_or_crop, load_only_fake):
    # Real data loader
    image_key = 'image'

    original_data_dir = "../../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256"
    # Load Training and Test data split
    with open("../../../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json", 'r') as f:
        data_split = json.load(f)
        training = data_split['training']

    # Loading all real cases to memory
    real_image_files = []
    for file_name in training:
        if 'empty' not in file_name['seg']:
            real_image_files.append(file_name['image'])
    print(f"real_image_files: {len(real_image_files)}")
        
    fake_image_files = []
    black_list = []
    for idx, img in enumerate(real_image_files):
        fake_name = img.split('/')[-1].replace('.nii.gz', '_0000.nii.gz')
        fake_name = f"synt_{fake_name}"
        fake_path = os.path.join(fake_data_dir, fake_name)
 
        if os.path.exists(fake_path):
            fake_image_files.append(fake_path)
        else:
            black_list.append(img.split('/')[-1])
            real_image_files.pop(idx)
    print(f"fake_image_files: {len(fake_image_files)}")
    print(f"real_image_files: {len(real_image_files)}")
    data_list = [{"image": img} for img in fake_image_files]
    print(f"fake data_list: {data_list[0]}")
    fake_data_loader = get_data_loader(image_key, clip_min, clip_max, data_list, pad_or_crop=pad_or_crop, two_loaders=False, cache_rate=[0])
    
    if not load_only_fake:       
        original_data_list = [{"image": os.path.join(original_data_dir, img)} for img in real_image_files]
        print(f"real original_data_list: {original_data_list[0]}") 
        real_data_loader = get_data_loader(image_key, clip_min, clip_max, original_data_list, pad_or_crop=pad_or_crop, two_loaders=False, cache_rate=[0]) # change 0 to 1
        return real_data_loader, fake_data_loader, black_list
    else:
        return fake_data_loader, black_list
    
def compute_MAE_folder_Tumour_GANs(clip_min, clip_max, load_only_fake, pad_or_crop, fake_data_dir):
    real_data_loader, fake_data_loader, black_list = get_CT_dataloader_GANs(clip_min, clip_max, fake_data_dir, pad_or_crop=pad_or_crop, load_only_fake=load_only_fake)
    real_iter = iter(real_data_loader)
    mae_results = {}
    for fake_batch_idx, fake_batch in enumerate(fake_data_loader): # iterate over fake cases 
        real_batch = next(real_iter)
        case_name = fake_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].replace('_0000.nii.gz', '.nii.gz')
        case_name = case_name.replace('synt_','')
        mae_here = 0
        if case_name in black_list:
            print(f"case_name {case_name} in black list.")
            continue
        else:
            y_fake = fake_batch['image']
            y_real = real_batch['image']
            while case_name != real_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]:
                real_batch = next(real_iter)
                warnings.warn(f"The cases don't match. Double check: {case_name} \n {real_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]}", SyntaxWarning)
            try:
                try:
                    mae_here = absolute_mean_error(y_real, y_fake)
                except:
                    mae_here = absolute_mean_error(y_real[:,:,:-1,:-1,:-1], y_fake)
            except:
                try:
                    # Some cases don't have the same orientation. Therefore, they need to be flipped
                    y_fake = y_fake.permute(0, 1, 2, 4, 3)
                    y_fake = torch.flip(y_fake, dims=[4])
                    mae_here = absolute_mean_error(y_real, y_fake)
                except:
                    # Some cases don't have the same orientation. Therefore, they need to be flipped
                    y_fake = y_fake.permute(0, 1, 2, 4, 3)
                    y_fake = torch.flip(y_fake, dims=[4])
                    mae_here = absolute_mean_error(y_real[:,:,:-1,:-1,:-1], y_fake)
            #pbar.set_postfix({"mae_here": float(mae_here)})
            mae_results[case_name.replace('.nii.gz', '')] = float(mae_here)
            print(f"mae_here: {mae_here}")
            
    print(f"Done: {mae_results}")
    json_file_path = os.path.join("./metrics/MAE", 
                                  fake_data_dir.split('nnUNet_raw/')[-1].split('imagesTr')[0], 'MAE.json')
    
    os.makedirs(json_file_path.replace('MAE.json', ''), exist_ok=True)
    
    with open(json_file_path, "w") as json_file:
        json.dump(mae_results, json_file, indent=4)

## Run for the GANs datasets
fake_data_dir = "../../nnUNet/nnUNet_raw/Dataset983_HnN_GAN/imagesTr"
clip_min=-200
clip_max=200
print(f"Doing {fake_data_dir}")
compute_MAE_folder_Tumour_GANs(clip_min, clip_max, load_only_fake, pad_or_crop, fake_data_dir)




