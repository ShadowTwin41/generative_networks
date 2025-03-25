# general settings
#GPU=1;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
MODE='c_sample';             # train vs sample / c_train vs c_sample
TRAIN_MODE=concat_cond;     # Default, conv_before_concat, concat_cond (new default, where the contrast is concat directly without convolution or wavelet)
DATASET=hnn;          # hnn or c_brats (for conditional brats) or hnn_tumour_inpainting
MODEL='ours_unet_256';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'

OUT_CHANNEL=8;

EMPTY_SEG=False; # True if I don't want to use the segmentation as condition
TUMOUR_WEIGHT=0; # 0 if EMPTY_SEG=True. Can be different if TRAIN_MODE=tumour_inpainting.
USE_LABEL_COND=True;
USE_LABEL_COND_CONV=False; # True if conv_before_concat, else False
# If USE_LABEL_COND_CONV is False, LABEL_COND_IN_CHANNELS is meaningless
LABEL_COND_IN_CHANNELS=0 # Number of channels of the condition used as input (hnn-> no_contrast/contrast/label; brats->three_label_channel)
                        # This is used for the convolution before concat with the image transformed with the wavelet.

FULL_BACKGROUND=False; # TODO double check
USE_WAVELET=True;

if [[ $MODE == 'c_train' ]]; then
  ROI_DATAAUG=True; # TODO -> done in the 29 oct 2024 
  IN_CHANNEL=16;
  if [[ $USE_LABEL_COND_CONV == 'False' ]]; then
    IN_CHANNEL=10;
  fi
elif [[ $MODE == 'c_sample' ]]; then
  ROI_DATAAUG=False;
  IN_CHANNEL=16;
  if [[ $USE_LABEL_COND_CONV == 'False' ]]; then
    IN_CHANNEL=10;
  fi
else
  IN_CHANNEL=8;
fi

echo IN_CHANNEL=${IN_CHANNEL};

# settings for sampling/inference
ITERATIONS=6600;             # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=100;         # number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="runs/hnn_CT_concat_cond__data_augment_7_11_2024_18:20:10/";               # tensorboard dir to be set for the evaluation # Most recente "runs/hnn_CT_24_8_2024_13:59:14"

OUTPUT_DIR=./results/${RUN_DIR}/dpm_plus_plus

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
    DATA_DIR=.utils/hnn.csv; # This will be the csv file 
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
    DATA_DIR=.utils/hnn.csv; # This will be the csv file 
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
    DATA_DIR=.utils/hnn.csv; # This will be the csv file 
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
fi

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
--ROI_DataAug=${ROI_DATAAUG}
"
TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=5000
--tumour_weight=${TUMOUR_WEIGHT}
--data_seg_augment=False
--label_cond_noise=False 
--empty_seg=${EMPTY_SEG}
--full_background=${FULL_BACKGROUND}
--modality=CT
--clip_min=-200
--clip_max=200
--train_mode=${TRAIN_MODE}
"
# data_seg_augment -> Old version
# label_cond_noise -> old version

# when tumour_weight > 0 the tumour loss is added, which means that the it is added a mse loss for the tumour voxels
# when label_cond_noise=True, it adds the same noise that was used to add into the input image to the segmentation.
#--devices=${GPU}
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
--empty_seg=${EMPTY_SEG}
--full_background=${FULL_BACKGROUND}
--modality=CT
--clip_min=-200
--clip_max=200
--mode=${MODE}
--train_mode=${TRAIN_MODE}
"
#--devices=${GPU}

echo "${DATA_DIR}";

# run the python scripts

python scripts/ct_scan_generator.py $SAMPLE $COMMON;

