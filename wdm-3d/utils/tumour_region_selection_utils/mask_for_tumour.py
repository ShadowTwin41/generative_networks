import os
import subprocess
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_closing
from skimage.morphology import disk, ball

def connect_structures_with_large_closing(segmentation, iterations=5, closing_radius=3, dilation_structure=None):
    """
    Applies a larger morphological closing to structures with value 1 using a custom structuring element, 
    then connects structures with value 2 by dilation while preserving structures with value 1.

    Parameters:
    - segmentation: np.ndarray, the input segmentation with values 1 and 2.
    - iterations: int, number of dilation iterations to connect structures.
    - closing_radius: int, radius for structuring element used in closing (controls hole size).
    - dilation_structure: np.ndarray, structuring element for dilation (optional).

    Returns:
    - np.ndarray: segmentation with closed structures for label 1 and connected structures for label 2.
    """
    # Step 1: Morphologically close the structures labeled with 1 using a larger structuring element
    structure_1_mask = segmentation == 1
    
    closing_structure = ball(closing_radius)  # 3D ball

    # Apply morphological closing to label 1 structures
    closed_structure_1 = binary_closing(structure_1_mask, structure=closing_structure)
    
    # Update the segmentation to reflect closed structures for label 1
    closed_segmentation = segmentation.copy()
    closed_segmentation[closed_structure_1] = 1

    # Step 2: Dilation loop to connect structures with value 2
    structure_2_mask = closed_segmentation == 2
    connected_structure = structure_2_mask.copy()
    
    for _ in range(iterations):
        # Dilate the `2` structures
        connected_structure = binary_dilation(connected_structure, structure=dilation_structure)
        # Apply the mask to avoid touching `1` structures
        connected_structure = np.logical_and(connected_structure, ~closed_structure_1)

    # Step 3: Update the segmentation with connected `2` structures
    connected_segmentation = closed_segmentation.copy()
    connected_segmentation[connected_structure] = 2

    return connected_segmentation

def connect_and_restrict_dilation(segmentation, ct_scan, iterations=20, dilation_structure=None, threshold=-190):
    """
    Dilates structures with value 2 while avoiding areas with CT values below a threshold and
    preserving structures with value 1.

    Parameters:
    - segmentation: np.ndarray, the input segmentation with values 1 and 2.
    - ct_scan: np.ndarray, the corresponding CT scan with intensity values.
    - iterations: int, number of dilation iterations to connect structures.
    - dilation_structure: np.ndarray, structuring element for dilation (optional).
    - threshold: int, threshold value for CT scan (default: -190).

    Returns:
    - np.ndarray: segmentation with label 2 dilated within CT and segmentation constraints.
    """
    # Step 1: Prepare masks
    structure_2_mask = segmentation == 2
    structure_1_mask = segmentation == 1
    
    # CT mask where values are >= threshold
    ct_mask = ct_scan >= threshold

    # Step 2: Dilation loop to connect structures with value 2, constrained by CT mask and structure_1_mask
    dilated_structure_2 = structure_2_mask.copy()
    for _ in range(iterations):
        # Dilate the `2` structures
        dilated_structure_2 = binary_dilation(dilated_structure_2, structure=dilation_structure)
        # Apply masks to limit expansion
        dilated_structure_2 = np.logical_and(dilated_structure_2, ct_mask)   # Respect CT values
        dilated_structure_2 = np.logical_and(dilated_structure_2, ~structure_1_mask)  # Avoid label 1 areas

    # Step 3: Update the segmentation with the dilated label 2 structures
    constrained_segmentation = segmentation.copy()
    constrained_segmentation[dilated_structure_2] = 2

    return constrained_segmentation

def set_not_two_to_one(segmentation_array):
    """
    Sets all values in the segmentation array that are not equal to 2 to 1.
    
    Parameters:
    - segmentation_array: numpy array (2D, 3D, or higher) of the segmentation data.
    
    Returns:
    - modified_array: numpy array with values set as specified.
    """
    # Create a copy of the original array to avoid modifying it in place
    modified_array = np.where(segmentation_array == 2, 2, 1)
    return modified_array

def get_paths(totalseg_input, totalseg_output):
    """
    Return the path for structures to avoid and to have tumour.
    Arguments:
        totalseg_input: Path to the CT scan used as input of the TotalSegmentor
        totalseg_output: Path with the mode folders from the TotalSegmentor output
    Returns:
        case_paths: CT scan path.
        case_paths_to_avoid: Path for segmentation of structures to avoid.
        case_paths_for_tumour: Path for segmentation of structures to have the tumour.
    """
    search_list_to_have_tumour = [ # From 1 up to 12
        "hypopharynx.nii.gz",
        "nasal_cavity_left.nii.gz",
        "nasal_cavity_right.nii.gz",
        "nasopharynx.nii.gz",
        "oropharynx.nii.gz",
        "soft_palate.nii.gz",
        "cricoid_cartilage.nii.gz",
        "larynx_air.nii.gz",
        "thyroid_cartilage.nii.gz",
        "trachea.nii.gz",
        "thyroid_gland.nii.gz",
        "esophagus.nii.gz",
        "hard_palate.nii.gz",
        "parotid_gland_left.nii.gz",
        "parotid_gland_right.nii.gz",
        "submandibular_gland_left.nii.gz",
        "submandibular_gland_right.nii.gz",
        "tongue.nii.gz",
        ]
    search_list_to_avoid_tumour = [ # From 13 up to 32
        # Places to avoid
        "lung_upper_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "spinal_cord.nii.gz",
        "skull.nii.gz",
        "brain.nii.gz",
        'vertebrae_C1.nii.gz', 
        'vertebrae_C2.nii.gz', 
        'vertebrae_C3.nii.gz', 
        'vertebrae_C4.nii.gz', 
        'vertebrae_C5.nii.gz', 
        'vertebrae_C6.nii.gz', 
        'vertebrae_C7.nii.gz',
        'vertebrae_T1.nii.gz', 
        'vertebrae_T2.nii.gz', 
        'vertebrae_T3.nii.gz', 
        'vertebrae_T4.nii.gz', 
        'vertebrae_T5.nii.gz', 
        'vertebrae_T6.nii.gz', 
        'vertebrae_T7.nii.gz',
        'sternum.nii.gz',
        "eye_left.nii.gz",
        "eye_right.nii.gz",
        "eye_lens_left.nii.gz",
        "eye_lens_right.nii.gz",
        "hyoid.nii.gz",
        "rib_left_1.nii.gz",
        "rib_left_2.nii.gz",
        "rib_left_3.nii.gz",
        "rib_left_4.nii.gz",
        "rib_left_5.nii.gz",
        "rib_left_6.nii.gz",
        "rib_right_1.nii.gz",
        "rib_right_2.nii.gz",
        "rib_right_3.nii.gz",
        "rib_right_4.nii.gz",
        "rib_right_5.nii.gz",
        "rib_right_6.nii.gz",
        "optic_nerve_left.nii.gz",
        "optic_nerve_right.nii.gz",
    ]

    case_name = totalseg_input.split(".nii.gz")[0].split("/")[-1]
    case_paths_for_tumour = {}
    case_paths_to_avoid = {}
    case_paths = {}
    
    # set the CT scan
    case_paths["CT_scan"] = os.path.join(totalseg_input)
    for structure in search_list_to_have_tumour: # structure -> what segmented structure
        FLAG_FOUND = False
        for modes in os.listdir(totalseg_output): # modes used to make the segmentation
            if not modes.endswith("nii.gz") and modes!="post_processed":
                complete_path = os.path.join(totalseg_output, modes, case_name, structure)
                if os.path.exists(complete_path):
                    case_paths_for_tumour[structure.split(".nii.gz")[0]] = os.path.join(complete_path)
                    FLAG_FOUND = True
                    break
        if not FLAG_FOUND:
            print(f"Not Found {structure}")
            print("______________")

    for structure in search_list_to_avoid_tumour: # structure -> what segmented structure
        FLAG_FOUND = False
        for modes in os.listdir(totalseg_output): # modes used to make the segmentation
            if not modes.endswith("nii.gz") and modes!="post_processed":
                complete_path = os.path.join(totalseg_output, modes, case_name, structure)
                if os.path.exists(complete_path):
                    case_paths_to_avoid[structure.split(".nii.gz")[0]] = os.path.join(complete_path)
                    FLAG_FOUND = True
                    break
        if not FLAG_FOUND:
            print(f"Not Found {structure}")
            print("______________")
    return case_paths, case_paths_to_avoid, case_paths_for_tumour

def run_total_segmentator(totalseg_input, totalseg_output):
    tasks_list = [
    "total",
    #"total_v1", -> not available
    ##"oculomotor_muscles",
    ## "face", -> model weights not open 
    "head_glands_cavities",
    "headneck_bones_vessels",
    "head_muscles",
    ##"class_map_part_organs"
    ]
    
    
    file_name = totalseg_input.split("/")[-1]
    for task in tasks_list:
        OUT_FILE = os.path.join(totalseg_output, task, file_name.split(".nii.gz")[0])
        print(f"OUT_FILE: {OUT_FILE}")
        if os.path.exists(OUT_FILE):
            print(f"Skipping Task {task} for {file_name}.")
        else:
            # Construct the command as a list
            command = ["TotalSegmentator", "-i", totalseg_input, "-o", OUT_FILE, "--task", task]
            # Run the command
            try:
                
                print(f"Running the command {command}")
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print("Command output:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("An error occurred:", e.stderr)

def create_mask_for_tumour(totalseg_input, totalseg_output):
    """
    Saves the mask for the region to have tumour (with value 2) and region to avoid (with value 1).
    Arguments:
        totalseg_input: Path to the CT scan used as input of the TotalSegmentor
        totalseg_output: Path with the mode folders from the TotalSegmentor output
    Returns:
        None
    """
    # Create the segmentations using the Total Segmentator
    run_total_segmentator(totalseg_input, totalseg_output)
    # Create the post-processed segmentation
    #os.makedirs(f"{totalseg_output}/post_processed", exist_ok=True)
    case_paths, case_paths_to_avoid, case_paths_for_tumour = get_paths(totalseg_input=totalseg_input, totalseg_output=totalseg_output)
    print(f"case_paths: {case_paths}")
    print(f"Creating label")

    case_name = totalseg_input.split(".nii.gz")[0].split("/")[-1]
    
    """
    # Value 1 corresponds to place where no tumour can be places, ever
    # Value 2 corresponds to place where a tumour can be placed
    """
    structure_paths_for_tumour = case_paths_for_tumour
    structure_paths_to_avoid = case_paths_to_avoid

    # Start by defining the bone (place where no tumour exists)
    final_segmentation_image = nib.load(case_paths["CT_scan"])
    final_segmentation = np.copy(final_segmentation_image.get_fdata())
    final_segmentation[final_segmentation < 180] = 0
    final_segmentation[final_segmentation >= 180] = 1

    # Make sure all structures are segmented by including the pre-segmented regions to not include tumour
    for structure_to_avoid in structure_paths_to_avoid:
        nii_image = nib.load(structure_paths_to_avoid[structure_to_avoid])
        nii_data = nii_image.get_fdata()
        final_segmentation[nii_data > 0] = 1

    # Overwrite the previous segmentation with places we know that can have tumour
    for structure_for_tumour in structure_paths_for_tumour:
        nii_image = nib.load(structure_paths_for_tumour[structure_for_tumour])
        nii_data = nii_image.get_fdata()
        final_segmentation[nii_data > 0] = 2
    final_segmentation_for_tumour = np.copy(final_segmentation)

    # Dillation of the region to contain tumour
    final_segmentation = connect_structures_with_large_closing(segmentation=final_segmentation, iterations=5, closing_radius=5, dilation_structure=None)
    
    # Huge dillation of the reagion to contain a tumour, avoiding the air regions that are not already segmented (like airway)
    final_segmentation = connect_and_restrict_dilation(segmentation=final_segmentation, ct_scan=np.copy(final_segmentation_image.get_fdata()), iterations=40, dilation_structure=None, threshold=-190)

    # Everythying that is not 2 at this point should not have tumour
    final_segmentation = set_not_two_to_one(segmentation_array=final_segmentation)
            
    # Overwrite the previous segmentation with places we know that can have tumour
    final_segmentation[final_segmentation_for_tumour==2] = 2
        
    # Save the final segmentation to a NIfTI file
    final_segmentation_nifti = nib.Nifti1Image(final_segmentation, affine=final_segmentation_image.affine, header=final_segmentation_image.header)
    output_path = f"{totalseg_output}/{case_name}_tumour_place.nii.gz"
    nib.save(final_segmentation_nifti, output_path)
    print(f"Final segmentation saved as {output_path}")
    
if __name__ == "__main__":
    files_path = "/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/runs/hnn_CT_concat_cond__data_augment_7_11_2024_18:20:10/Correct_naming"
    for file_name in os.listdir(files_path):
        if "generated" in file_name:
            totalseg_input = f"/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/runs/hnn_CT_concat_cond__data_augment_7_11_2024_18:20:10/Correct_naming/{file_name}"
        totalseg_output = files_path
        create_mask_for_tumour(totalseg_input, totalseg_output)