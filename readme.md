# Generation of Synthetic Datasets Will Solve Anonymisation Problem for Collaborative Medical Image Analysis and Data Sharing: Feasibility Demonstrated in Head and Neck CT Images and Brain Tumour MRI Images

A brief description of what this project does and who it's for.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/yourusername/yourproject.git

# Navigate to the project directory
cd yourproject

conda create -n wdm python=3.10.13 -y
conda activate wdm

conda install -c conda-forge \
    numpy=1.26.4 \
    pywavelets=1.4.1 \
    scipy=1.12.0 \
    pip -y

pip install \
    nibabel==5.2.0 \
    blobfile==2.1.1 \
    tensorboard==2.16.2 \
    matplotlib==3.8.3 \
    tqdm==4.66.2 \
    dicom2nifti==2.4.10

Install PyTorch from https://pytorch.org/get-started/locally/
pip install monai
pip install PyWavelets
```

## Run conditional training with WDM
cd wdm-3d
change the run.sh file. Ensure that:
    MODE='c_train'
    TRAIN_MODE='conv_before_concat', 'concat_cond' or 'wavelet_cond'


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.