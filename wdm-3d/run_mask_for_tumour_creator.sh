#!/usr/bin/env sh
# This is a sample script to execute python scripts on the HPC cluster using a conda environment.
# CUDA will be loaded.

# This script must be submitted from a Rocky8 login node
# You can submit it via "sbatch job.sh" and cancel via "scancel <job_id>"
# View the queue status via: "squeue -u $USER --start"

#SBATCH --partition=GPUampere
#SBATCH --time=480:00:00
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#### bash run.sh
##  FAKE_CAES_DIR=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Bone_segmentation/200/Original_1000
##  OUT_FILE=/projects/brats2023_a_f/Aachen/aritifcial-head-and-neck-cts/WDM3D/wdm-3d/results/Synthetic_Datasets/Whole_scans/Bone_segmentation/Mask_for_tumour_inpaint/200/Original_1000
FAKE_CAES_DIR=$1
OUT_FILE=$2
echo "FAKE_CAES_DIR=$FAKE_CAES_DIR"
echo "OUT_FILE=$OUT_FILE"
python utils/run_mask_for_tumour_creator.py $FAKE_CAES_DIR $OUT_FILE

#### SBATCH --account=rwth1484

#### # SBATCH --job-name=WDM_with_label_augment_full_background
#### # SBATCH --output=output.WDM_with_label_augment_full_background_%J.txt

#### # SBATCH --job-name=WDM_empty_label_full_background
#### # SBATCH --output=output.WDM_empty_label_full_background_%J.txt

#### # SBATCH --job-name=WDM_no_seg_but_full_background
#### # SBATCH --output=output.WDM_no_seg_but_full_background_%J.txt

#### # SBATCH --job-name=HNN_inpaint_tumour
#### # SBATCH --output=output.HNN_inpaint_tumour_%J.txt


