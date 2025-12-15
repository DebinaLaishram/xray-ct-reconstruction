import os
import subprocess

dicom_root = "/Users/debinalaishram/xrayct_project/data/dicom_raw"
nifti_root = "/Users/debinalaishram/xrayct_project/data/nifti"

os.makedirs(nifti_root, exist_ok=True)

patients = sorted(os.listdir(dicom_root))

for patient in patients:
    dicom_dir = os.path.join(dicom_root, patient)
    output_file = os.path.join(nifti_root, f"{patient}.nii.gz")
    
    # Skip if already converted
    if os.path.exists(output_file):
        print(f"Skipping {patient}, already exists.")
        continue

    # Call the single-patient conversion script
    cmd = [
        "python",
        "/Users/debinalaishram/xrayct_project/src/preprocessing/convert_dicom_to_nifti.py",
        "--input", dicom_dir,
        "--output", output_file
    ]
    print(f"Converting {patient}...")
    subprocess.run(cmd)

