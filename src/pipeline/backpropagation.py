"""
backprojection.py

Implements Eq. (1): Back-projection of AP (0°) X-ray images
into a rough 3D volume (Vbp).

AP (160 x 160)  -->  Vbp (160 x 160 x 128)

Notes:
- Uses only data/{train,val,test}/AP as input
- Casts AP to float32
- Normalization: divide by D only (Δp implicit)
- Generates Vbp for train / val / test
"""

import os
import numpy as np
from tqdm import tqdm


# -------------------------
# Configuration (LOCKED)
# -------------------------

# Project root = xrayct_project/
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

AP_SHAPE = (160, 160)
DEPTH = 128  # D

SPLITS = ["train", "val", "test"]

AP_DIRNAME = "AP"
VBP_ROOT = "data/vbp"


# -------------------------
# Back-projection function
# -------------------------

def backproject_ap_to_vbp(ap_2d: np.ndarray, depth: int) -> np.ndarray:
    """
    Back-project a single AP image into a rough 3D volume.

    Parameters
    ----------
    ap_2d : np.ndarray
        AP image of shape (160, 160)
    depth : int
        Number of depth slices (128)

    Returns
    -------
    vbp : np.ndarray
        Back-projected volume of shape (160, 160, 128), float32
    """
    assert ap_2d.shape == AP_SHAPE, (
        f"Expected AP shape {AP_SHAPE}, got {ap_2d.shape}"
    )

    # Cast to float32
    ap_2d = ap_2d.astype(np.float32)
    
    # Eq. (1) back-projection:
    # V_bp(x,y,z) = I_AP(x,y) / |L|
    # Note: Δp (voxel spacing) is implicit since CTs were resampled
    # to fixed 1.5 mm isotropic spacing and intensities normalized.


    # Eq. (1): distribute intensity along depth
    ap_2d = ap_2d / depth

    # Expand into 3D volume (H, W, D)
    vbp = np.repeat(ap_2d[:, :, None], depth, axis=2)

    return vbp


# -------------------------
# Process one split
# -------------------------

def process_split(split: str):
    ap_dir = os.path.join(PROJECT_ROOT, f"data/{split}/{AP_DIRNAME}")
    vbp_dir = os.path.join(PROJECT_ROOT, VBP_ROOT, split)

    if not os.path.isdir(ap_dir):
        raise FileNotFoundError(f"AP directory not found: {ap_dir}")

    os.makedirs(vbp_dir, exist_ok=True)

    ap_files = sorted(
        f for f in os.listdir(ap_dir) if f.endswith(".npy")
    )

    print(f"\n[{split.upper()}]")
    print(f"AP input directory : {ap_dir}")
    print(f"Vbp output directory: {vbp_dir}")
    print(f"Number of AP files : {len(ap_files)}")

    for fname in tqdm(ap_files, desc=f"{split}"):
        ap_path = os.path.join(ap_dir, fname)
        vbp_path = os.path.join(vbp_dir, fname)

        ap = np.load(ap_path)

        vbp = backproject_ap_to_vbp(ap, DEPTH)

        # Safety checks
        assert vbp.shape == (AP_SHAPE[0], AP_SHAPE[1], DEPTH)
        assert vbp.dtype == np.float32

        np.save(vbp_path, vbp)


# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    print("Starting AP → Vbp back-projection")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"AP shape: {AP_SHAPE}, Depth: {DEPTH}")

    for split in SPLITS:
        process_split(split)

    print("\nBack-projection completed successfully.")
