import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class VbpToCTDataset(Dataset):
    """
    Dataset for:
        Vbp (rough 3D volume)  -->  CT_gt (ground-truth CT)

    Splits are defined by:
        data/vbp/{train,val,test}/

    Ground-truth CTs are stored in:
        data/nifti_resampled/

    Matching is done strictly by LIDC-ID.
    """

    def __init__(self, root_dir, split):
        """
        Args:
            root_dir (str): project root directory (xrayct_project)
            split (str): one of ['train', 'val', 'test']
        """
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.root_dir = root_dir
        self.split = split

        self.vbp_dir = os.path.join(root_dir, "data", "vbp", split)
        self.ct_dir = os.path.join(root_dir, "data", "nifti_resampled")

        assert os.path.isdir(self.vbp_dir), f"Vbp directory not found: {self.vbp_dir}"
        assert os.path.isdir(self.ct_dir), f"CT directory not found: {self.ct_dir}"

        # Collect Vbp files
        self.vbp_files = sorted(
            [f for f in os.listdir(self.vbp_dir) if f.endswith("_AP.npy")]
        )

        assert len(self.vbp_files) > 0, f"No Vbp files found in {self.vbp_dir}"

        print(
            f"[Dataset] Split: {split} | "
            f"Samples: {len(self.vbp_files)}"
        )

    def __len__(self):
        return len(self.vbp_files)

    def __getitem__(self, idx):
        # -----------------------------
        # Load Vbp
        # -----------------------------
        vbp_filename = self.vbp_files[idx]
        vbp_path = os.path.join(self.vbp_dir, vbp_filename)

        vbp = np.load(vbp_path).astype(np.float32)

        # Expected shape: (160, 160, 128)
        assert vbp.ndim == 3, f"Vbp must be 3D, got {vbp.shape}"

        # -----------------------------
        # Extract case ID
        # -----------------------------
        # Example: LIDC-IDRI-0001_AP.npy → LIDC-IDRI-0001
        case_id = vbp_filename.replace("_AP.npy", "")

        # -----------------------------
        # Load GT CT
        # -----------------------------
        ct_filename = f"{case_id}.nii.gz"
        ct_path = os.path.join(self.ct_dir, ct_filename)

        assert os.path.isfile(ct_path), f"CT file not found: {ct_path}"

        ct = nib.load(ct_path).get_fdata().astype(np.float32)

        # -----------------------------
        # Safety checks
        # -----------------------------
        assert ct.shape == vbp.shape, (
            f"Shape mismatch for {case_id}: "
            f"Vbp {vbp.shape} vs CT {ct.shape}"
        )

        # -----------------------------
        # Convert to torch tensors
        # -----------------------------
        # Start as (1, H, W, D)
        vbp = torch.from_numpy(vbp).unsqueeze(0)
        ct = torch.from_numpy(ct).unsqueeze(0)

        # Permute to (C, D, H, W) for Conv3d
        vbp = vbp.permute(0, 3, 1, 2)  # (1, 128, 160, 160)
        ct = ct.permute(0, 3, 1, 2)

        return {
            "vbp": vbp,
            "ct": ct,
            "id": case_id,
        }
