# generate_projections.py

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Paths
input_dir = "/Users/debinalaishram/xrayct_project/data/nifti_resampled/"
output_dir = "/Users/debinalaishram/xrayct_project/data/projections/"

# Create subfolders for AP and LAT
ap_dir = os.path.join(output_dir, "AP")
lat_dir = os.path.join(output_dir, "LAT")
os.makedirs(ap_dir, exist_ok=True)
os.makedirs(lat_dir, exist_ok=True)

# Projection function
def generate_projections(volume, method='mean'):
    """
    volume: 3D numpy array
    method: 'max', 'mean', or 'sum'
    Returns: AP and LAT projections
    """
    if method == 'max':
        ap_proj = np.max(volume, axis=2)
        lat_proj = np.max(volume, axis=0)
    elif method == 'mean':
        ap_proj = np.mean(volume, axis=2)
        lat_proj = np.mean(volume, axis=0)
    elif method == 'sum':
        ap_proj = np.sum(volume, axis=2)
        lat_proj = np.sum(volume, axis=0)
    else:
        raise ValueError("method must be 'max', 'mean', or 'sum'")
    return ap_proj, lat_proj


# Scale projection for visualization
def scale_for_display(proj):
    proj_min = proj.min()
    proj_max = proj.max()
    if proj_max - proj_min > 0:
        proj_scaled = (proj - proj_min) / (proj_max - proj_min)
    else:
        proj_scaled = proj
    return proj_scaled

# Loop over all CT volumes
for fname in os.listdir(input_dir):
    if not fname.endswith(".nii.gz"):
        continue

    ct_path = os.path.join(input_dir, fname)
    nii = nib.load(ct_path)
    vol = nii.get_fdata()

    # Generate projections
    ap_proj, lat_proj = generate_projections(vol)

    # Save projections as .npy (network input/output)
    base_name = fname.replace(".nii.gz", "")
    np.save(os.path.join(ap_dir, f"{base_name}_AP.npy"), ap_proj)
    np.save(os.path.join(lat_dir, f"{base_name}_LAT.npy"), lat_proj)

    # Save projections as .png for visual check (scaled)
    plt.imsave(os.path.join(ap_dir, f"{base_name}_AP.png"), scale_for_display(ap_proj), cmap='gray')
    plt.imsave(os.path.join(lat_dir, f"{base_name}_LAT.png"), scale_for_display(lat_proj), cmap='gray')

    print(f"Generated AP/Lat projections for {fname}")

print("All projections generated.")
