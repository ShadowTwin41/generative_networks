## Compute the MS-SSIM metric
ROOT_DIR="../results/Synthetic_Datasets/";
DATASET=$1; # TODO change here


CLIP_MIN=$2;
CLIP_MAX=$3;
JSON_FILE_PATH=./metrics/MS-SSIM/$4
echo "CLIP_MIN=$CLIP_MIN"
echo "CLIP_MAX=$CLIP_MAX"
echo "JSON_FILE_PATH=$JSON_FILE_PATH"

DATA_DIR="$ROOT_DIR$DATASET"
echo "DATA_DIR=$DATA_DIR"

python 3_0_Compute_MS-SSIM_CT.py $DATA_DIR $CLIP_MIN $CLIP_MAX $JSON_FILE_PATH

## bash run_MSSSIM_CT.sh "Whole_scans/Tumour_inpaint/full_blur/200/Original_1000"

