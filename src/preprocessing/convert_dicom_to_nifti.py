import os
import pydicom
import numpy as np
import nibabel as nib


def load_dicom_series(dicom_dir):
    """Load a folder (and subfolders) of DICOM slices into a 3D numpy array."""
    dicoms = []

    for root, _, files in os.walk(dicom_dir):
        for fname in files:
            if fname.lower().endswith(".dcm"):
                path = os.path.join(root, fname)
                dicoms.append(pydicom.dcmread(path))

    if len(dicoms) == 0:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # Sort slices by InstanceNumber
    dicoms.sort(key=lambda x: int(x.InstanceNumber))

    # Stack pixel arrays
    volume = np.stack([d.pixel_array for d in dicoms], axis=-1)

    # Get voxel spacing
    pixel_spacing = dicoms[0].PixelSpacing
    slice_thickness = dicoms[0].SliceThickness
    spacing = np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness])

    return volume, spacing

def convert_to_nifti(dicom_dir, output_path):
    volume, spacing = load_dicom_series(dicom_dir)
    affine = np.diag([*spacing, 1])
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, output_path)
    print(f"Saved NIfTI to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of DICOM slices")
    parser.add_argument("--output", required=True, help="Output .nii.gz file")
    args = parser.parse_args()
    convert_to_nifti(args.input, args.output)
