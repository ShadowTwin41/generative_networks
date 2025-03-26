# Generation of Synthetic Datasets Will Solve Anonymisation Problem for Collaborative Medical Image Analysis and Data Sharing: Feasibility Demonstrated in Head and Neck CT Images and Brain Tumour MRI Images


## Table of Contents

- [Installation](#installation)
- [Run conditional training with WDM](#run-conditional-training-with-wdm)
- [nnUNet - Segmentation](#nnunet---segmentation)
- [Ground truth bone](#ground-truth-bone)
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

## Run conditional training with WDM
```cd wdm-3d```

To start the training with the full resolution scans:
  *   In the run.sh (for the CT dataset), or in run_brats.sh (for the MRI dataset) file. Ensure that:
      * MODE='c_train' # For training
        * To resume the training: ```--resume_checkpoint='...' --resume_step=...```

      * MODE='c_sample' # For inference
        * ```ITERATIONS=...; SAMPLING_STEPS=1000; RUN_DIR="runs/..."; OUTPUT_DIR=./results/...; ```
      * TRAIN_MODE='conv_before_concat', 'concat_cond' or 'wavelet_cond'
      * --save_interval=100 # Adjust this value to your machine

  * $M_{all\_conv}^{WDM_{200}}$ or $M_{seg\_conv}^{WDM_{MRI}}$:
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

  * $M_{all\_d}^{WDM_{200}}$ or $M_{seg\_d}^{WDM_{MRI}}$:
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

  * $M_{all\_w}^{WDM_{200}}$ or $M_{seg\_w}^{WDM_{MRI}}$:
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

  * $M_{ROI\_d}^{WDM_{200}}$:
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

  * $M_{ROI\_d}^{WDM_{1000}}$:
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

To start the training of the inpainting model:
  *   In the run_inpaint.sh Ensure that:
      * MODE='c_train' # For training
        * To resume the training: ```--resume_checkpoint='...' --resume_step=...```

      * MODE='c_sample' # For inference
        * ```ITERATIONS=...; SAMPLING_STEPS=1000; RUN_DIR="runs/..."; OUTPUT_DIR=./results/...; ```
      * --save_interval=100 # Adjust this value to your machine
  
  * $M_{all\_cat}^{DDPM_{200}}$:
```
TRAIN_MODE=default_tumour_inpainting; 
DATASET=hnn_tumour_inpainting; 
USE_WAVELET=False;
--clip_min=-200;
--clip_max=200;
```

  * $M_{all\_cat}^{DDPM_{1000}}$:
```
TRAIN_MODE=default_tumour_inpainting; 
DATASET=hnn_tumour_inpainting; 
USE_WAVELET=False;
--clip_min=-1000;
--clip_max=1000;
```

* For inference
  * ```TRAIN_MODE=default_tumour_inpainting; BLUR_MASK=None; FROM_MONAI_LOADER=True; ``` # To generate cropped volumes
  * ```TRAIN_MODE=tumour_inpainting; BLUR_MASK='edge_blur' or 'full_blur' OUTPUT_DIR='./results/...'; INPUT_DIR='./results/...';``` # To inpaint in full resolution scan 


* For inference with ```scheduler_list = ["DPM++_2M", "DPM++_2M_Karras", "DPM++_2M_SDE", "DPM++_2M_SDE_Karras"]```
  * Make the changes in ```scripts/infer_DPM++_models_CT.py``` or ```scripts/infer_DPM++_models_MRI.py``` or ```scripts/infer_DPM++_inpaint_models.py``` and run it.
    * Mainly ```pretrained_weights_path```
  * For inpainting a tumour on full resolution scans, use ```run_inder_DPM++_inpaint.sh``` 
    * Change: ```CLIP_MIN=; CLIP_MAX=; MODEL_PATH=; USE_MASK_BLUR=; output_dir=; data_dir=;```

## nnUNet - Segmentation
We saved the version installed of the nnUNet, which is available in the nnUNet folder. 

No major changes were made to it, therefore, installing the nnUNet from the [source](https://github.com/MIC-DKFZ/nnUNet) might lead to the same or better results/performance. 
## Ground truth bone:
The following repositories were used to create the ground truth for bone segmentation. 
Follow the steps in each of these:
* [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

* [totalspineseg](https://github.com/neuropoly/totalspineseg)

* [AMASSS_CBCT](https://github.com/Maxlo24/AMASSS_CBCT)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.