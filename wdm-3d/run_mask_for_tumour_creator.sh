# It creates the mask to ensure that the tumour is not generated in a wrong place

#### examples
##  FAKE_CAES_DIR=./results/Synthetic_Datasets/Whole_scans/Bone_segmentation/200/Original_1000
##  OUT_FILE=./results/Synthetic_Datasets/Whole_scans/Bone_segmentation/Mask_for_tumour_inpaint/200/Original_1000
FAKE_CAES_DIR=$1
OUT_FILE=$2
echo "FAKE_CAES_DIR=$FAKE_CAES_DIR"
echo "OUT_FILE=$OUT_FILE"
python utils/run_mask_for_tumour_creator.py $FAKE_CAES_DIR $OUT_FILE


