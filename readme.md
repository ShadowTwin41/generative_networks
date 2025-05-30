# Enhancing Privacy: The Utility of Stand-Alone Synthetic CT and MRI for Tumor and Bone Segmentation


## Table of Contents
- [Enhancing Privacy: The Utility of Stand-Alone Synthetic CT and MRI for Tumor and Bone Segmentation](#enhancing-privacy-the-utility-of-stand-alone-synthetic-ct-and-mri-for-tumor-and-bone-segmentation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [GANs training](#gans-training)
  - [Run conditional training with WDM](#run-conditional-training-with-wdm)
    - [Configurations for training each model (Full resolution)](#configurations-for-training-each-model-full-resolution)
    - [Configurations for training each model (inpainting models)](#configurations-for-training-each-model-inpainting-models)
      - [For inference](#for-inference)
    - [DPM++ Inference](#dpm-inference)
  - [nnUNet - Segmentation](#nnunet---segmentation)
  - [Ground truth bone](#ground-truth-bone)
  - [Evaluation metrics](#evaluation-metrics)
  - [License](#license)

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/ShadowTwin41/generative_networks.git

# Navigate to the project directory
cd generative_networks

conda create -n wdm python=3.10.13 -y
conda activate wdm

conda install -c conda-forge \
    numpy=1.26.4 \
    scipy=1.12.0 \
    pip -y

pip install \
    nilearn \
    nibabel==5.2.0 \
    blobfile==2.1.1 \
    tensorboard==2.16.2 \
    matplotlib==3.8.3 \
    tqdm==4.66.2 \
    dicom2nifti==2.4.10 \
    scikit-image \
    diffusers["torch"] \
    transformers \
    monai \
    PyWavelets \
    pandas

Install PyTorch from https://pytorch.org/get-started/locally/
pip install TotalSegmentator
```
 ## GANs training 
 1️⃣ -> ```cd GANs```
 * It requires two GPUs to train (one for the Generator and one for the Discriminator)
 * To train for the CT in src/run
   * ```python cWGAN_GP_style_256.py  --W_ADV_D 1 --W_ADV_G 1 --W_PWA 1000 --W_PWT 100 --W_GP 10 --IN_CHANNEL_G 3 --OUT_CHANNEL_G 1 --IN_CHANNEL_D 4  --LR_D 0.0002 --LR_G 0.0002 --TOTAL_EPOCHS 1000 --NUM_WORKERS 6 --DATASET training --UNET Unet_FC --SKIP_LATENT False --TAHN_ACT False --DA True --NORM_FUNC Linear --CLIP_MIN -200 --CLIP_MAX 200 --EXP_NAME W_PWA1000__W_PWT100__Unet_FC_min200_200 --RESUME 117```
 
 * To train Brats in src/run
   * ```python cWGAN_GP_style_256_BraTS.py  --W_ADV_D 1 --W_ADV_G 1 --W_PWA 100 --W_PWT 100 --W_GP 10 --IN_CHANNEL_G 3 --OUT_CHANNEL_G 1 --IN_CHANNEL_D 4  --LR_D 0.0002 --LR_G 0.0002 --TOTAL_EPOCHS 1000 --NUM_WORKERS 6 --DATASET training --UNET Unet_FC --SKIP_LATENT False --TAHN_ACT False --DA True --EXP_NAME BraTS_W_PWA100__W_PWT100__Unet_FC_new --RESUME 990```
 
 * src/notebooks contains the codes for inference.  The script in src/run /CT_HNC_synthetic_generation can also be used for it.
 
  * For inference:
    * ```src/notebooks```  
    * The script in ```src/run /CT_HNC_synthetic_generation.py``` can also be used.
  
## Run conditional training with WDM
1️⃣ -> ```cd wdm-3d```

💻 To start the training with the full resolution scans:
  *   In the run.sh (for the CT dataset), or in run_brats.sh (for the MRI dataset) file. Ensure that:
      * ```MODE='c_train'``` # For training
        * To resume the training: ```--resume_checkpoint='...' --resume_step=...```

      * MODE='c_sample' # For inference
        * ```ITERATIONS=...; SAMPLING_STEPS=1000; RUN_DIR="runs/..."; OUTPUT_DIR=./results/...; ```
      * ```TRAIN_MODE='conv_before_concat', 'concat_cond' or 'wavelet_cond'```
      * ```--save_interval=100``` # Adjust this value to your machine

### Configurations for training each model (Full resolution)
 * ❗ -> for the CT, use the file ```run.sh```
 * ❗ -> for the MRI, use the file ```run_brats.sh``` 

 ⚠️ -> For training ```MODE='c_train'``` | For inference ```MODE='c_sample'```
  * $WDM_{all\_conv}^{200}$ or $WDM_{seg\_conv}^{MRI}$:
```
TRAIN_MODE='conv_before_concat';
NO_SEG=False;
TUMOUR_WEIGHT=0;
REMOVE_TUMOUR_FROM_LOSS=False;
USE_LABEL_COND=True
USE_LABEL_COND_CONV=True;
LABEL_COND_IN_CHANNELS=3
FULL_BACKGROUND=False
USE_WAVELET=True;
--clip_min=-200;
--clip_max=200;
```

  * $WDM_{all\_d}^{200}$ or $WDM_{seg\_d}^{MRI}$:
```
TRAIN_MODE='concat_cond';
NO_SEG=False;
TUMOUR_WEIGHT=0;
REMOVE_TUMOUR_FROM_LOSS=False;
USE_LABEL_COND=True
USE_LABEL_COND_CONV=False;
LABEL_COND_IN_CHANNELS=0
FULL_BACKGROUND=False
USE_WAVELET=True;
--clip_min=-200;
--clip_max=200;
```

  * $WDM_{all\_w}^{200}$ or $WDM_{seg\_w}^{MRI}$:
```
TRAIN_MODE='wavelet_cond';
NO_SEG=False;
TUMOUR_WEIGHT=0;
REMOVE_TUMOUR_FROM_LOSS=False;
USE_LABEL_COND=True
USE_LABEL_COND_CONV=False;
LABEL_COND_IN_CHANNELS=0
FULL_BACKGROUND=False
USE_WAVELET=True;
--clip_min=-200;
--clip_max=200;
```

  * $WDM_{ROI\_d}^{200}$:
```
TRAIN_MODE='concat_cond';
NO_SEG=True;
TUMOUR_WEIGHT=0;
REMOVE_TUMOUR_FROM_LOSS=False;
USE_LABEL_COND=True
USE_LABEL_COND_CONV=False;
LABEL_COND_IN_CHANNELS=0
FULL_BACKGROUND=False
USE_WAVELET=True;
--clip_min=-200;
--clip_max=200;
```

  * $WDM_{ROI\_d}^{1000}$:
```
TRAIN_MODE='concat_cond';
NO_SEG=True;
TUMOUR_WEIGHT=0;
REMOVE_TUMOUR_FROM_LOSS=False;
USE_LABEL_COND=True
USE_LABEL_COND_CONV=False;
LABEL_COND_IN_CHANNELS=0
FULL_BACKGROUND=False
USE_WAVELET=True;
--clip_min=-1000;
--clip_max=1000;
```
### Configurations for training each model (inpainting models)
  *   In the ```run_inpaint.sh``` Ensure that:
      * MODE='c_train' # For training
        * To resume the training: ```--resume_checkpoint='...' --resume_step=...```

      * MODE='c_sample' # For inference
        * ```ITERATIONS=...; SAMPLING_STEPS=1000; RUN_DIR="runs/..."; OUTPUT_DIR=./results/...; ```
      * ```--save_interval=100``` # Adjust this value to your machine
  
  * $DDPM_{all\_cat}^{200}$:
```
TRAIN_MODE=default_tumour_inpainting; 
DATASET=hnn_tumour_inpainting; 
USE_WAVELET=False;
--clip_min=-200;
--clip_max=200;
```

  * $DDPM_{all\_cat}^{1000}$:
```
TRAIN_MODE=default_tumour_inpainting; 
DATASET=hnn_tumour_inpainting; 
USE_WAVELET=False;
--clip_min=-1000;
--clip_max=1000;
```

#### For inference
  * ```TRAIN_MODE=default_tumour_inpainting; BLUR_MASK=None; FROM_MONAI_LOADER=True; ``` # To generate cropped volumes
  * ```TRAIN_MODE=tumour_inpainting; BLUR_MASK='edge_blur' or 'full_blur'; FROM_MONAI_LOADER=False; OUTPUT_DIR='./results/...'; INPUT_DIR='./results/...';``` # To inpaint in full resolution scan. Do not forget to create the masks for the region to inpaint, using ```run_mask_for_tumour_creator.sh```. The TotalSegmentator might need a distinct PyTorch version.

### DPM++ Inference
* For inference with ```scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]```
  * Make the changes in (mainly ```pretrained_weights_path```):
    *  ```scripts/infer_DPM++_models_CT.py``` -> for full resolution CT;
    *  ```scripts/infer_DPM++_models_MRI.py``` -> for full resolution MRI; 
    *  ```scripts/infer_DPM++_inpaint_models.py``` -> for the cropped volumes;
    *  ```scripts/infer_DPM++_tumour_inpainting.py``` -> for the inpaint of tumor on full resolution scans. 
  
  * For inpainting a tumour on full resolution scans, use ```run_infer_DPM++_inpaint.sh``` 
    * Change: ```CLIP_MIN=; CLIP_MAX=; MODEL_PATH=; USE_MASK_BLUR=; output_dir=; data_dir=;```

## nnUNet - Segmentation
We saved the version installed of the nnUNet, which is available in the nnUNet folder. 

No major changes were made to it, therefore, installing the nnUNet from the [source](https://github.com/MIC-DKFZ/nnUNet) might lead to the same or better results/performance. 
## Ground truth bone
The following repositories were used to create the ground truth for bone segmentation. 
Follow the steps in each of these:
* [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

* [totalspineseg](https://github.com/neuropoly/totalspineseg)

* [AMASSS_CBCT](https://github.com/Maxlo24/AMASSS_CBCT)

## Evaluation metrics
❗-> These files might not be adapted to your use case. Analyse them before runing.
In the folder wdm-3d/notebooks are the files used for computing the MAE, MS-SSIM and the Radiomics.

* The files ```Radiomics_tumour.py``` ```Radiomics_soft_tissue.py``` and ```Radiomics_bone.py``` can be used to extract the features of the tumour region, soft tissue and bone, respectively. Adapt this files to your case. 

* For data mining, use the  ```Radiomics.ipynb``` or ```Radiomics_MRI.ipynb``` notebooks. These files might be changed! No guaranties are given.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
