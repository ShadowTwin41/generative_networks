ROOT_DIR="../results/Synthetic_Datasets/";
DATASET=$1; # TODO change here

DATA_DIR="$ROOT_DIR$DATASET"
JSON_FILE_PATH=./metrics/MS-SSIM/$4
echo "DATA_DIR=$DATA_DIR"
echo "JSON_FILE_PATH=$JSON_FILE_PATH"

python Compute_MS-SSIM_MRI.py $DATA_DIR $JSON_FILE_PATH

## Example:
## bash run_MSSSIM_MRI.sh MRI/Tumour_generation/wavelet_cond/DPM++_2M_SDE_Karras MRI/Tumour_generation/wavelet_cond/DPM++_2M_SDE_Karras/MS-SSIM.json