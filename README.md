# X-ray → CT Reconstruction Pipeline

This repository implements an X-ray to CT reconstruction pipeline inspired by  
**DVG-Diffusion: Dual-View Guided Diffusion Model for CT Reconstruction from X-Rays**  
https://arxiv.org/pdf/2503.17804

The current implementation focuses on reconstructing a **3D CT volume from a single AP (0°) X-ray**, followed by **lateral (90°) view synthesis** via forward projection.

## Data Source

This project uses publicly available CT scans from the **LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)** dataset.

- Source: The Cancer Imaging Archive (TCIA)  
- Dataset: LIDC-IDRI  
- Modality: Chest CT  
- Data format: DICOM (converted to NIfTI during preprocessing)

LIDC-IDRI provides thoracic CT scans with standardized acquisition protocols and is widely used for research in medical image analysis. All preprocessing, projection generation, and dataset splitting were performed as part of this project.

## Repository Structure

```text
xrayct_project/
├── src/
│ ├── preprocessing/        # DICOM → NIfTI, resampling, projections
│ └── pipeline/             # Training, inference, forward projection, evaluation
│
├── data/                   # Dataset structure (raw data not included)
│ ├── dicom_raw/            # Raw DICOMs (excluded)
│ ├── nifti_resampled/      # Standardized CT volumes (excluded)
│ ├── projections/
│ │ ├── AP/
│ │ └── LAT/
│ ├── train/
│ │ ├── AP/
│ │ └── LAT/
│ ├── val/
│ │ ├── AP/
│ │ └── LAT/
│ ├── test/
│ │ ├── AP/
│ │ └── LAT/
│ └── vbp/
│   ├── train/
│   ├── val/
│   └── test/
│
├── runs/
│ └── ct_refine/
│   ├── checkpoints/        # Trained model weights
│   ├── debug_slices/       # Debug visualizations (PNG)
│   └── test_predictions/   # Prediction visualizations (PNG only)
│
├── results/
│ ├── metrics/
│ ├── examples/
│ └── summary.txt
│
├── README.md
└── .gitignore


**Note:** Raw medical imaging data (DICOM, NIfTI, NumPy volumes) is not redistributed and must be obtained directly from TCIA (LIDC-IDRI).


The pipeline consists of:
1. Standardized CT preprocessing  
2. Physics-inspired back-projection (Equation 1)  
3. AP and LAT ground-truth projection generation  
4. Dataset splitting into Train / Validation / Test  
5. A 3D U-Net for CT refinement  
6. Forward projection to obtain predicted LAT view (Equation 9)  
7. Quantitative evaluation of predicted LAT using PSNR and SSIM  

---

## CT Preprocessing Pipeline

### 1. DICOM → NIfTI Conversion
Raw LIDC-IDRI CT scans are provided in DICOM format. These are converted to NIfTI for easier
processing and compatibility with common medical imaging libraries.

- Script: `src/preprocessing/convert_dicom_to_nifti.py`  
- Output: NIfTI CT volumes  

---

### 2. Resampling and Standardization
To ensure consistent spatial resolution and intensity distribution across subjects, all CT
volumes are standardized using the following steps:

- Resample to **1.5 mm isotropic voxel spacing**
- Pad or crop volumes to a fixed shape: **160 × 160 × 128**
- Clip Hounsfield Units (HU) to **[-1000, 400]**
- Normalize intensities to **[0, 1]**
- Convert data type to `float32`

The resulting volumes serve as **ground-truth CT** during training.

> **Note:** After resampling and normalization, all subsequent processing operates on voxel
indices. Physical spacing does not explicitly appear in later equations.

---

### 3. AP / LAT Projection Generation
From each standardized CT volume, two X-ray projections are generated:

- **AP (0°)** projection  
- **LAT (90°)** projection  

These projections are used for:
- Training input (AP)
- Evaluation and future supervision (LAT)

Projections are stored as:
- `.npy` files for model input/output
- `.png` files for visualization and sanity checks

---

### 4. Dataset Splitting
Subjects are split into mutually exclusive sets:

- **Training:** 70%  
- **Validation:** 15%  
- **Test:** 15%  

AP and LAT projections are organized consistently across splits to prevent data leakage.

---

## Back-Projection for Rough 3D Volume Construction (Equation 1)

Direct reconstruction of a 3D CT volume from a single 2D X-ray is ill-posed, as depth information
is lost during projection.

To reduce the learning difficulty, a **rough 3D volume** is first constructed by back-projecting
the AP X-ray into the CT voxel space.

---

### Notation

- I_in ∈ R^{H × W}: input AP (0°) X-ray  
- V_bp ∈ R^{H × W × D}: back-projected 3D volume  
- (x, y): pixel coordinates  
- z: depth index  
- D: number of depth slices

---

### Back-Projection Operator

The back-projected volume is defined as:
V_bp = BP(I_in)

For an AP acquisition, each pixel value is replicated uniformly along the depth dimension:
V_bp(x, y, z) ∝ I_in(x, y), for all z ∈ {1, …, D}

This simplified formulation conveys the idea of depth-wise back-projection; the full operator definition,
including geometric constraints and normalization, follows Eq. 1 in the original paper.

---

### Practical Interpretation

As a result:
- All depth slices initially appear similar
- No anatomical structure is recovered at this stage

This behavior is expected and confirms the correctness of the back-projection.

---

### Purpose of the Rough 3D Volume

The back-projected volume V_bp:

- Is **not** a valid CT reconstruction  
- Acts as a **geometrically aligned 3D scaffold**  
- Bridges 2D X-ray space and 3D CT space  
- Converts an ill-posed 2D→3D problem into a **3D→3D refinement task**

---

## Dataset Preparation

Each training sample consists of:

- **Input:** rough 3D volume V_bp
- **Target:** ground-truth CT volume  

Samples are matched strictly by **LIDC-ID**.

The dataset returns tensors in the format:

`(C, D, H, W) = (1, 128, 160, 160)`


This format is directly compatible with PyTorch `Conv3d`.

---

## 3D U-Net for CT Refinement

A 3D U-Net is trained to map the rough 3D volume to a refined CT volume.

- **Input:** rough 3D volume V_bp
- **Output:** refined 3D CT volume  

Architecture:
- Encoder–decoder with skip connections  
- 3D convolutions  
- Group Normalization  

The network learns to infer missing anatomical structure while preserving geometric consistency.

---

## Training

- Script: `src/pipeline/train_ct.py`  
- Objective:
  - L1 loss between predicted CT and ground-truth CT  
- Optimizer:
  - Adam  
- Device support:
  - Apple M-series (MPS), CUDA, or CPU  

Outputs include:
- Model checkpoints
- Debug visualizations comparing:
  - Back-projected volume
  - Ground-truth CT
  - Predicted CT  

---

## Forward Projection and LAT View Synthesis (Equation 9)

After CT refinement, a lateral (90°) X-ray view is synthesized via forward projection:

I_LAT(y, z) = ∑ₓ V_CT(x, y, z)

This expression illustrates the principle of forward projection for a canonical lateral view; the full
geometry-aware formulation follows Eq. 9 in the original paper using a differentiable CT-to-X-ray projector.

- Script: `src/pipeline/forwardprojection.py`  
- Output: predicted LAT view stored as `.npy`

---

## Evaluation

Predicted LAT views are evaluated against ground-truth LAT projections using:

- **PSNR**
- **SSIM**

- Script: `src/pipeline/evaluation.py`  
- Metrics are saved to:
`results/metrics/metrics_all.csv`

- Summary statistics are saved to:
`results/summary.txt`


Qualitative examples (AP → predicted LAT → ground-truth LAT) for all test cases are exported to:

`results/examples/`

---

## Results Summary

On the held-out test set, the developed pipeline achieves:

- **PSNR ≈ 22.6**
- **SSIM ≈ 0.83**

These results are competitive with and exceed the baseline performance reported in Table VI of the reference paper for
0° → 90° new view synthesis.
