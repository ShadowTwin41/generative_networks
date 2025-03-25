#!/bin/bash
#SBATCH --partition=GPUampere
#SBATCH --time=480:00:00
#SBATCH --job-name=infer_tumour_inpainting_real_case_DPM_edge_blur_200_n0
#SBATCH --output=infer_tumour_inpainting_real_case_DPM_edge_blur_200_n0_%J.txt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

### TO change!
CLIP_MIN=-200;
CLIP_MAX=200;
MODEL_PATH=../aritifcial-head-and-neck-cts/WDM3D/wdm-3d/runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__data_augment_20_11_2024_11:07:31/checkpoints/hnn_tumour_inpainting_2000000.pt

## CLIP_MIN=-1000;
## CLIP_MAX=1000
## MODEL_PATH=../aritifcial-head-and-neck-cts/WDM3D/wdm-3d/runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_18_12_2024_14:25:08/checkpoints/hnn_tumour_inpainting_2000000.pt
 
USE_MASK_BLUR=edge_blur ## edge_blur or full_blur
### TO change!

# --output_dir=../aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Tumour_inpaint/$USE_MASK_BLUR/$CLIP_MAX
# --data_dir=../aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Bone_segmentation/$CLIP_MAX
SAMPLE="
--model_path=$MODEL_PATH
--output_dir=../aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Tumour_inpaint/$USE_MASK_BLUR/$CLIP_MAX/realCT_fakeTumour
--data_dir=../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256
--json_file=../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json
--batch_size=1
--num_workers=8
--renormalize=None
--empty_seg=True
--full_background=True
--clip_min=$CLIP_MIN
--clip_max=$CLIP_MAX
--use_wavelet=False
--image_size=128
--use_dilation=True
--use_data_augmentation=False
--modality=CT
--train_mode=tumour_inpainting
--mode=c_sample
--dataset=hnn_tumour_inpainting
--sampling_steps=100
--use_mask_blur=$USE_MASK_BLUR
"

# run the python scripts
#python scripts/tumour_inpainting_same_case.py $SAMPLE; # change --output_dir=

python scripts/tumour_inpainting.py $SAMPLE; # change --output_dir=

#--data_dir=../HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/data_split.json