import os
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
from os import listdir
from os.path import join
import sys
import nibabel as nib
import numpy as np
from tqdm import tqdm



# Initialize Pyradiomics feature extractor
params_file = "./metrics/RADIOMICS/uitls/Parameters.yml"  # Optional YAML file for feature selection
extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

def save_results_to_csv(feature_data, output_csv):
    if not feature_data:
        print("No features extracted!")
        return

    # Extract the feature keys from the first result
    keys = feature_data[0].keys()

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(feature_data)
    print(f"Results saved to {output_csv}")


# Helper function to extract features
def extract_features(image_mask_pair):
    ct_path, mask_path = image_mask_pair
    print(f"Doing: {ct_path.name}")
    result = extractor.execute(str(ct_path), str(mask_path))
    return {ct_path.name: result}

from concurrent.futures import ProcessPoolExecutor

def run_parallel_extraction(image_mask_list, output_csv):
    # Submit each extraction job along with its associated input case
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [(item, executor.submit(extract_features, item)) for item in image_mask_list]

    feature_data = []
    for item, future in futures:
        try:
            result = future.result()
        except Exception as e:
            print(f"Error processing case {item}: {e}")
            continue
        
        for filename, features in result.items():
            features["Filename"] = filename
            feature_data.append(features)

    save_results_to_csv(feature_data, output_csv)



def create_binary_seg(labelsTr, nnunet_id, datasets_save_folder):
    save_folder = join(datasets_save_folder, nnunet_id, 'labelsTr')
    os.makedirs(save_folder, exist_ok=True)
    print(f"{save_folder} created")
    if len(listdir(save_folder))==len(listdir(labelsTr)):
        print(f"{save_folder} already done")
        return None
    for file_name in tqdm(listdir(labelsTr)):
        input_path = join(labelsTr, file_name)
        output_path = join(datasets_save_folder, nnunet_id, 'labelsTr', file_name)

        # Load the segmentation file
        segmentation = nib.load(input_path)
        segmentation_data = segmentation.get_fdata()

        # Convert to binary mask, ignoring labels 16 and 31
        binary_mask = np.where((segmentation_data != 16) & (segmentation_data != 31) & (segmentation_data != 0), 1, 0)

        # Create a new NIfTI image with the binary mask
        binary_mask_img = nib.Nifti1Image(binary_mask, segmentation.affine, segmentation.header)

        # Save the binary mask to the output path
        nib.save(binary_mask_img, output_path)


nnUNet_raw = "../../nnUNet/nnUNet_raw"
metrics_radiomics_folder = "./metrics/RADIOMICS"

nnunet_id = str(sys.argv[1])
imagesTr = join(nnUNet_raw, nnunet_id, 'imagesTr_clipped_1000')
labelsTr = join(nnUNet_raw, nnunet_id, 'labelsTr')

datasets_save_folder = "./metrics/RADIOMICS/uitls/Datasets"
binary_labelsTr = join(datasets_save_folder, nnunet_id, 'labelsTr')

# Run the Binary mask creator
create_binary_seg(labelsTr, nnunet_id, datasets_save_folder)

# Gather image-mask pairs
image_mask_list = [
    (ct_path, Path(binary_labelsTr) / ct_path.name.replace('_0000',''))
    for ct_path in Path(imagesTr).glob("*.nii.gz")
    if (Path(binary_labelsTr) / ct_path.name.replace('_0000','')).exists() 
]
print(image_mask_list[0])
print(len(image_mask_list))
output_folder = join(metrics_radiomics_folder, nnunet_id)
os.makedirs(output_folder, exist_ok=True)
output_csv = join(output_folder, "bone_radiomics.csv")
run_parallel_extraction(image_mask_list, output_csv)