# general settings
#GPU=1;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
MODE='c_sample';             # train vs sample / c_train vs c_sample
TRAIN_MODE=tumour_inpainting;     # Default, tumour_inpainting (to replace the healthy region generated with tumour), conv_before_concat, concat_cond, default_tumour_inpainting (trained giving the seg as condition, but noise in all scan)
# TODO: TRAIN_MODE needs to be tumour_inpainting for the always known case
USE_DILATION=True;                # True to create and use a version dilated of the segmentation (in the data loader)
DATASET=hnn_tumour_inpainting;          # hnn or c_brats (for conditional brats) or hnn_tumour_inpainting
MODEL='ours_unet_128';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'

USE_WAVELET=False;

if [[ $TRAIN_MODE == 'tumour_inpainting' ]]; then
  MODEL=ours_unet_128;
  IN_CHANNEL=4;
  OUT_CHANNEL=1;
  USE_LABEL_COND=True;
  USE_LABEL_COND_CONV=False;
  LABEL_COND_IN_CHANNELS=0
  USE_WAVELET=False;
  USE_DILATION=False;   #This is True mainly for the always known version  # TODO for training
  TUMOUR_WEIGHT=10; # This is mainly for the always known version (set to 10) # TODO for training
  USE_DATA_AUGMENTATION=False; # This is True mainly for the default version # TODO for training
  if [[ $MODE == 'c_sample' ]]; then
    BLUR_MASK=full_blur; # edge_blur for full noise on label tumour / full_blur for blur on all dilated tumour (some background information is present on the place of the tumour)
    USE_DATA_AUGMENTATION=False; # 
  fi
  if [[ $USE_DILATION == 'True' ]]; then
    IN_CHANNEL=5
  fi
fi

if [[ $TRAIN_MODE == 'default_tumour_inpainting' ]]; then
  MODEL=ours_unet_128;
  IN_CHANNEL=4;
  OUT_CHANNEL=1;
  USE_LABEL_COND=True;
  USE_LABEL_COND_CONV=False;
  LABEL_COND_IN_CHANNELS=0
  USE_WAVELET=False;
  USE_DILATION=False;  
  TUMOUR_WEIGHT=10; 
  USE_DATA_AUGMENTATION=True;
  if [[ $MODE == 'c_sample' ]]; then
    BLUR_MASK=full_blur; # edge_blur for full noise on label tumour / full_blur for blur on all dilated tumour (some background information is present on the place of the tumour)
    USE_DATA_AUGMENTATION=True; # 
  fi
  if [[ $USE_DILATION == 'True' ]]; then
    IN_CHANNEL=5
  fi
fi

echo IN_CHANNEL=${IN_CHANNEL};

# settings for sampling/inference
ITERATIONS=2000;             # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=1000;         # number of steps for accelerated sampling, 0 for the default 1000
## 200
RUN_DIR="runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__data_augment_20_11_2024_11:07:31/";               # tensorboard dir to be set for the evaluation
OUTPUT_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Tumour_inpaint/with_mask/full_blur/200/Original_1000
INPUT_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Bone_segmentation/200/Original_1000
#OUTPUT_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Tumour_inpaint/edge_blur/200/realCT_fakeTumour # For real dataset
#INPUT_DIR=/projects/brats2023_a_f/Aachen/HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256/ # For real dataset
FROM_MONAI_LOADER=False;

## 1000
## RUN_DIR="runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_18_12_2024_14:25:08/";
## OUTPUT_DIR="/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Tumour_inpaint/$BLUR_MASK/1000/Original_1000";

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "MODEL: WDM (U-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1; # it was 10
elif [[ $MODEL == 'ours_unet_256' ]]; then
  echo "MODEL: WDM (U-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "MODEL: WDM (WavU-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=10;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "MODEL: WDM (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

# some information and overwriting batch size for sampling
# (overwrite in case you want to sample with a higher batch size)
# no need to change for reproducing

#/work/bl256603/HnN_cancer/HnN_cancer_data/HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256
#/projects/brats2023_a_f/Aachen/HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256
if [[ $MODE == 'sample' ]]; then
  echo "MODE: sample"
  BATCH_SIZE=1;
elif [[ $MODE == 'c_sample' ]]; then
  BATCH_SIZE=1;
  echo "MODE: c_sample"
  if [[ $DATASET == 'hnn' ]]; then
    echo "Dataset: Head and neck cancer";
    DATA_DIR=/projects/brats2023_a_f/Aachen/HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256;
  elif [[ $DATASET == 'c_brats' ]]; then
    echo "DATASET: Conditional BRATS";
    DATA_DIR=/projects/brats2023_a_f/BRAINTUMOUR/data/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData;
  elif [[ $DATASET == 'hnn_tumour_inpainting' ]]; then
    echo "DATASET: hnn_tumour_inpainting";
    DATA_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/utils/hnn.csv; # This will be the csv file 
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
elif [[ $MODE == 'train' ]]; then
  if [[ $DATASET == 'brats' ]]; then
    echo "MODE: training";
    echo "DATASET: BRATS";
    DATA_DIR=/projects/brats2023_a_f/BRAINTUMOUR/data/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData;
  elif [[ $DATASET == 'lidc-idri' ]]; then
    echo "MODE: training";
    echo "Dataset: LIDC-IDRI";
    DATA_DIR=~/wdm-3d/data/LIDC-IDRI/;
  elif [[ $DATASET == 'hnn' ]]; then
    echo "MODE: training";
    echo "Dataset: Head and neck cancer";
    DATA_DIR=/projects/brats2023_a_f/Aachen/HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256;
  elif [[ $DATASET == 'hnn_tumour_inpainting' ]]; then
    echo "DATASET: hnn_tumour_inpainting";
    DATA_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/utils/hnn.csv; # This will be the csv file 
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
elif [[ $MODE == 'c_train' ]]; then
  if [[ $DATASET == 'hnn' ]]; then
    echo "MODE: c_training";
    echo "Dataset: Head and neck cancer";
    DATA_DIR=/projects/brats2023_a_f/Aachen/HnN_cancer_data/HnN_cancer_data_1_1_1_256_256_256;
  elif [[ $DATASET == 'c_brats' ]]; then
    echo "MODE: training";
    echo "DATASET: Conditional BRATS";
    DATA_DIR=/projects/brats2023_a_f/BRAINTUMOUR/data/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData;
  elif [[ $DATASET == 'hnn_tumour_inpainting' ]]; then
    echo "DATASET: hnn_tumour_inpainting";
    DATA_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/utils/hnn.csv; # This will be the csv file 
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
fi

if [[ $USE_DATA_AUGMENTATION == 'True' ]]; then
  DATA_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/utils/hnn_DA.csv; 
  if [[ $MODE == 'c_sample' ]]; then
    DATA_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/utils/hnn.csv;
  fi
  echo "$DATA_DIR"
fi

# TODO diffusion_steps
COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNEL}
--out_channels=${OUT_CHANNEL}
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
--label_cond_in_channels=${LABEL_COND_IN_CHANNELS}
--num_workers=8
--use_label_cond=${USE_LABEL_COND}
--use_label_cond_conv=${USE_LABEL_COND_CONV}
--use_wavelet=${USE_WAVELET}
"
TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/runs/hnn_tumour_inpainting_CT_default_tumour_inpainting__DA_tumorW_10_28_11_2024_14:37:59/checkpoints/hnn_tumour_inpainting_1395000.pt
--resume_step=1395000
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=5000
--tumour_weight=${TUMOUR_WEIGHT}
--data_seg_augment=False
--label_cond_noise=False 
--full_background=True
--modality=CT
--clip_min=-200
--clip_max=200
--train_mode=${TRAIN_MODE}
--use_dilation=${USE_DILATION}
--use_data_augmentation=${USE_DATA_AUGMENTATION}
"
  
# when tumour_weight > 0 the tumour loss is added, which means that the it is added a mse loss for the tumour voxels
# when label_cond_noise=True, it adds the same noise that was used to add into the input image to the segmentation.
#--devices=${GPU}
# TODO diffusion_steps
SAMPLE="
--data_dir=${DATA_DIR}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=./${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt
--output_dir=${OUTPUT_DIR}
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
--full_background=True
--modality=CT
--clip_min=-200
--clip_max=200
--use_dilation=${USE_DILATION}
--use_data_augmentation=${USE_DATA_AUGMENTATION}
--train_mode=${TRAIN_MODE}
--blur_mask=${BLUR_MASK}
--from_monai_loader=${FROM_MONAI_LOADER}
--input_dir=${INPUT_DIR}
"
#--devices=${GPU}

echo "${DATA_DIR}";

# run the python scripts
if [[ $MODE == 'train' ]]; then
  python scripts/generation_train.py $TRAIN $COMMON;
elif [[ $MODE == 'c_train' ]]; then
  python scripts/generation_train_conditional.py $TRAIN $COMMON;
elif [[ $MODE == 'c_sample' ]]; then
  python scripts/generation_sample_conditional.py $SAMPLE $COMMON;
else
  python scripts/generation_sample.py $SAMPLE $COMMON;
fi
