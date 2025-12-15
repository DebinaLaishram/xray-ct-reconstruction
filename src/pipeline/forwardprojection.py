"""
Forward Projection (Eq. 9)
Convert refined CT volumes into LAT projections.

CT_pred: (1, z, y, x) or (z, y, x)
LAT_pred: (y, z)

No evaluation is done here.
"""

import os
import numpy as np

# --------------------
# Paths (LOCKED)
# --------------------
CT_PRED_DIR = "runs/ct_refine/test_predictions"
LAT_PRED_DIR = "runs/ct_refine/test_predictions"

# --------------------
# Utils
# --------------------
def load_ct_pred(path):
    vol = np.load(path)
    if vol.ndim == 4:          # (1, z, y, x)
        vol = vol[0]
    return vol.astype(np.float32)


def forward_project_lat(ct_vol):
    """
    ct_vol: (z, y, x)
    returns: (y, z)
    """
    lat_zy = ct_vol.sum(axis=2)   # sum over x → (z, y)
    lat_yz = lat_zy.T             # transpose → (y, z)
    return lat_yz


def normalize(img):
    return img / (img.max() + 1e-6)


# --------------------
# Main
# --------------------
def main():
    ct_files = sorted([
        f for f in os.listdir(CT_PRED_DIR)
        if f.endswith("_CT_pred.npy")
    ])

    if len(ct_files) == 0:
        raise RuntimeError("No CT prediction files found.")

    print(f"Found {len(ct_files)} CT prediction files.")
    print("Generating LAT projections...\n")

    for fname in ct_files:
        case_id = fname.replace("_CT_pred.npy", "")

        ct_path = os.path.join(CT_PRED_DIR, fname)
        lat_pred_path = os.path.join(
            LAT_PRED_DIR, f"{case_id}_LAT_pred.npy"
        )

        # Load CT
        ct_pred = load_ct_pred(ct_path)      # (z, y, x)

        # Forward projection
        lat_pred = forward_project_lat(ct_pred)  # (y, z)

        # Normalize
        lat_pred = normalize(lat_pred)

        # Save
        np.save(lat_pred_path, lat_pred)

        print(f"Saved: {case_id}_LAT_pred.npy  | shape={lat_pred.shape}")

    print("\n Forward projection complete.")


if __name__ == "__main__":
    main()
