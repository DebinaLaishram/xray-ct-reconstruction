import os
import nibabel as nib
import numpy as np
import pandas as pd

nifti_root = "/Users/debinalaishram/xrayct_project/data/nifti"
files = sorted([f for f in os.listdir(nifti_root) if f.endswith(".nii.gz")])

summary = []

for f in files:
    file_path = os.path.join(nifti_root, f)
    nii = nib.load(file_path)
    data = nii.get_fdata()
    
    info = {
        "file": f,
        "shape": data.shape,
        "slices": data.shape[2],
        "voxel_spacing_x": nii.header.get_zooms()[0],
        "voxel_spacing_y": nii.header.get_zooms()[1],
        "voxel_spacing_z": nii.header.get_zooms()[2],
        "dtype": data.dtype,
        "min": data.min(),
        "max": data.max(),
        "mean": data.mean(),
        "std": data.std()
    }
    summary.append(info)

# Convert to a DataFrame for easy analysis
df = pd.DataFrame(summary)

# Print overall statistics
print("Total files:", len(df))
print("Unique shapes:", df['shape'].unique())
print("Unique voxel spacings:", df[['voxel_spacing_x','voxel_spacing_y','voxel_spacing_z']].drop_duplicates().values)
print("Data types:", df['dtype'].unique())
print("Slices: min/max:", df['slices'].min(), "/", df['slices'].max())
print("Intensity ranges: min/max:", df['min'].min(), "/", df['max'].max())
print("Mean intensity range:", df['mean'].min(), "/", df['mean'].max())
print("Std dev range:", df['std'].min(), "/", df['std'].max())

# Optional: save summary to CSV for reference
df.to_csv(os.path.join(nifti_root, "nifti_summary.csv"), index=False)

