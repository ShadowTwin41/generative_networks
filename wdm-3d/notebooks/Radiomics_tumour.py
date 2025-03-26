import os
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
from os import listdir
from os.path import join
import sys
import nibabel as nib



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
    with ProcessPoolExecutor(max_workers=64) as executor:
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



def is_label_empty(label_path):
    """
    Check if the label file contains only zeros.

    Parameters:
    - label_path: Path to the label file.

    Returns:
    - True if the label file contains only zeros, False otherwise.
    """
    # Load the label file
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()

    # Check the shape of the label data
    if label_data.ndim < 2:
        raise ValueError(f"Mask has too few dimensions: {label_data.ndim}. Minimum required is 2.")

    # Check if the label data contains only zeros
    return not label_data.any()

nnUNet_raw = "../../nnUNet/nnUNet_raw"
metrics_radiomics_folder = "./notebooks/metrics/RADIOMICS"

nnunet_id = str(sys.argv[1])
imagesTr = join(nnUNet_raw, nnunet_id, 'imagesTr')
labelsTr = join(nnUNet_raw, nnunet_id, 'labelsTr')
# Gather image-mask pairs
image_mask_list = [
    (ct_path, Path(labelsTr) / ct_path.name.replace('_0000',''))
    for ct_path in Path(imagesTr).glob("*.nii.gz")
    if (Path(labelsTr) / ct_path.name.replace('_0000','')).exists() and
           not is_label_empty(Path(labelsTr) / ct_path.name.replace('_0000', ''))
]
print(image_mask_list[0])
print(len(image_mask_list))
output_folder = join(metrics_radiomics_folder, nnunet_id)
os.makedirs(output_folder, exist_ok=True)
output_csv = join(output_folder, "tumour_radiomics.csv")
run_parallel_extraction(image_mask_list, output_csv)