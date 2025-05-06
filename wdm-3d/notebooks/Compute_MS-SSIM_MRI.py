import argparse
import os
import numpy as np
import torch
import sys
import json
from glob import glob
from generative.metrics import MultiScaleSSIMMetric
from tqdm import tqdm
sys.path.append(".")
from test_utils import get_data_loader, get_BraTS_data_loader
from joblib import Parallel, delayed
import time

start_time = time.time()


def get_dataloaders_BraTS(data_dir):
    # Getting all files that end with _CT_n0.nii.gz from the data_dir
    image_files = sorted(glob(os.path.join(data_dir, "*t1c_n0.nii.gz"))) 
    if "Synthetic_Datasets" not in data_dir and "GAN" not in data_dir:
        files_names = sorted(glob(os.path.join(data_dir, "**", "*t1c.nii.gz"), recursive=True))  
        print(f"files_names: {files_names[0]}")
        # Loading the training cases
        with open('/projects/brats2023_a_f/BRAINTUMOUR/data/brats2023/BraTS2023_GLI_data_split.json', 'r') as file:
            data_training = json.load(file)
            data_training = data_training['training']

        allowed_cases = []
        for case in data_training:
            allowed_cases.append(case['t1c'].split('/')[-1])
        print(f"allowed_cases: {allowed_cases[0]}")
        
        for id_name in files_names:
            id_name = id_name.split('/')[-1]
            if id_name in allowed_cases:
                image_files.append(os.path.join(data_dir, id_name.split('-t1c.nii.gz')[0], id_name))
    elif "GAN" in data_dir:
        image_files = sorted(glob(os.path.join(data_dir, "*.nii.gz")))  

    data_list = [{"image": img} for img in image_files]
    print(f"data_list: {len(data_list)}")

    # It will load the dataloader_2 to memory -> dataloader_2 will iterate the same number of times as the dataloader_1 has of items
    dataloader_1 = get_BraTS_data_loader("image", data_list=data_list, cache_rate=0, apply_quantile=True) # TODO remove [:16]
    dataloader_2 = get_BraTS_data_loader("image", data_list=data_list, cache_rate=1, apply_quantile=True) # TODO cache_rate=1

    return dataloader_1, dataloader_2

def compute_ms_ssim(args):
    """Compute MS-SSIM between two images."""
    img1, img2, device = args
    ms_ssim_value = ms_ssim(img1.to(device), img2.to(device)).item()
    return ms_ssim_value

def main(dataloader_1, dataloader_2, device):
    ms_ssim_list = []
    for step_1, batch in enumerate(dataloader_1):
        print(f"Doing case {step_1}")
        img = batch['image']

        # Prepare arguments for multiprocessing
        args = []
        for step_2, batch2 in enumerate(dataloader_2):
            if step_1 == step_2:
                continue
            img2 = batch2['image']
            args.append((img, img2, device))
            if (step_2%32==0 or (step_2+1)==len(dataloader_2)) and step_2!=0:
                print(f"args: {len(args)}")
                # Use joblib to compute MS-SSIM with parallel processing
                results = Parallel(n_jobs=16)(
                    delayed(compute_ms_ssim)(arg) for arg in tqdm(args, desc="Computing MS-SSIM")
                )
                args = []
                print(f"results: {results}")
                ms_ssim_list.extend(results)
        print(f"ms_ssim_list: {len(ms_ssim_list)}")
    return ms_ssim_list

# Unchanged arguments
device = torch.device("cpu")
ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)

# Get each directory to compute the metric
data_dir = sys.argv[1]
json_file_path = sys.argv[2]

def run_BraTS_cases(data_dir):
    print("Runing a BraTS dataset")
    print(f"Getting dataloaders from {data_dir}")
    dataloader_1, dataloader_2 = get_dataloaders_BraTS(data_dir)
    print(f"Computing MS-SSIM: {data_dir} \n (this takes a while)...")
    ms_ssim_list = main(dataloader_1, dataloader_2, device)
    ms_ssim_list = np.array(ms_ssim_list)
    print("Calculated MS-SSIMs. Computing mean ...")
    print(f"Mean MS-SSIM: {ms_ssim_list.mean():.6f}")
    print(f"STD MS-SSIM: {ms_ssim_list.std():.6f}")
    mm_ssim = {}
    mm_ssim["mean"] = float(ms_ssim_list.mean())
    mm_ssim["std"] = float(ms_ssim_list.std())

    os.makedirs(json_file_path.replace('MS-SSIM.json', ''), exist_ok=True)

    with open(json_file_path, "w") as json_file:
        json.dump(mm_ssim, json_file, indent=4)

run_BraTS_cases(data_dir)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")