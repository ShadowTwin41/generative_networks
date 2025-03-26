# File to inpaint synthetic tumours using schedulers= ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]

### TO change!
CLIP_MIN=-200;
CLIP_MAX=200;
MODEL_PATH=./runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_25_3_2025_18:11:32/checkpoints/hnn_tumour_inpainting_001000.pt

## CLIP_MIN=-1000;
## CLIP_MAX=1000
## MODEL_PATH=./runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_25_3_2025_18:11:32/checkpoints/hnn_tumour_inpainting_001000.pt
 
USE_MASK_BLUR=edge_blur ## edge_blur or full_blur
### TO change!

SAMPLE="
--model_path=$MODEL_PATH
--output_dir=./results/Synthetic_Datasets/Whole_scans/Tumour_inpaint/$USE_MASK_BLUR/$CLIP_MAX
--data_dir=./results/Synthetic_Datasets/Whole_scans/Bone_segmentation/$CLIP_MAX
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
python scripts/infer_DPM++_tumour_inpainting.py $SAMPLE; # change --output_dir=

