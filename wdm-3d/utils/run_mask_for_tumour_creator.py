import os
import sys
sys.path.append(".")
from utils.tumour_region_selection_utils.mask_for_tumour import create_mask_for_tumour


FAKE_CAES_DIR = sys.argv[1]
OUT_FILE = sys.argv[2]
os.makedirs(OUT_FILE, exist_ok=True)

for file_name in os.listdir(FAKE_CAES_DIR):
    totalseg_input = os.path.join(FAKE_CAES_DIR, file_name)
    print(f"Doing {totalseg_input}")
    create_mask_for_tumour(totalseg_input=totalseg_input, totalseg_output=OUT_FILE)

