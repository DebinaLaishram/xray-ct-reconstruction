# resample_and_standardize.py

import os
import nibabel as nib
import numpy as np
import scipy.ndimage
import csv

# Paths
input_dir = "/Users/debinalaishram/xrayct_project/data/nifti/"
output_dir = "/Users/debinalaishram/xrayct_project/data/nifti_resampled/"
os.makedirs(output_dir, exist_ok=True)

# Parameters
target_spacing = (1.5, 1.5, 1.5)  # mm
target_shape = (160, 160, 128)    # H, W, Slices
hu_window = (-1000, 400)          # HU clipping window
normalize = True                   # Normalize to [0,1] after clipping

# CSV summary
csv_file = os.path.join(output_dir, "resample_summary.csv")
csv_header = ["filename", "orig_shape", "orig_spacing", "resampled_shape", "pad_crop_info"]

# Function to resample a volume to new spacing
def resample_volume(vol, orig_spacing, new_spacing):
    zoom_factors = [o/n for o, n in zip(orig_spacing, new_spacing)]
    return scipy.ndimage.zoom(vol, zoom=zoom_factors, order=1)  # linear interpolation

# Function to pad/crop to target shape
def pad_crop_volume(vol, target_shape):
    padded_vol = np.zeros(target_shape, dtype=vol.dtype)

    src_slices = []
    dst_slices = []
    pad_crop_info = ""

    for i in range(3):
        if vol.shape[i] < target_shape[i]:
            # Pad
            pad_before = (target_shape[i] - vol.shape[i]) // 2
            pad_after = pad_before + vol.shape[i]
            src_slices.append(slice(0, vol.shape[i]))
            dst_slices.append(slice(pad_before, pad_after))
            pad_crop_info += f"{i}:pad "
        else:
            # Crop
            crop_start = (vol.shape[i] - target_shape[i]) // 2
            crop_end = crop_start + target_shape[i]
            src_slices.append(slice(crop_start, crop_end))
            dst_slices.append(slice(0, target_shape[i]))
            pad_crop_info += f"{i}:crop "

    # Apply crop/pad
    padded_vol[dst_slices[0], dst_slices[1], dst_slices[2]] = vol[src_slices[0], src_slices[1], src_slices[2]]

    return padded_vol, pad_crop_info

# Remove old CSV if exists
if os.path.exists(csv_file):
    os.remove(csv_file)

# Process all NIfTIs
for fname in os.listdir(input_dir):
    if not fname.endswith(".nii.gz"):
        continue

    fpath = os.path.join(input_dir, fname)
    nii = nib.load(fpath)
    vol = nii.get_fdata()
    orig_spacing = nii.header.get_zooms()[:3]

    # Resample
    resampled_vol = resample_volume(vol, orig_spacing, target_spacing)
    resampled_shape = resampled_vol.shape

    # Pad/crop to target shape
    final_vol, pad_crop_info = pad_crop_volume(resampled_vol, target_shape)

    # Clip HU window
    final_vol = np.clip(final_vol, hu_window[0], hu_window[1])

    # Convert to float32
    final_vol = final_vol.astype(np.float32)

    # Optional normalization to [0,1]
    if normalize:
        final_vol = (final_vol - hu_window[0]) / (hu_window[1] - hu_window[0])

    # Save
    out_path = os.path.join(output_dir, fname)
    new_nii = nib.Nifti1Image(final_vol, affine=nii.affine)
    nib.save(new_nii, out_path)

    # Write summary
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(csv_header)
        writer.writerow([fname, vol.shape, orig_spacing, resampled_shape, pad_crop_info])

    print(f"Processed {fname}: resampled {resampled_shape} → target {target_shape}, {pad_crop_info}")

print("All volumes processed. Summary CSV saved at:", csv_file)
