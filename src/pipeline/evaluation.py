"""
Evaluation script for LAT view synthesis (0° → 90°).

Computes PSNR and SSIM between:
- Predicted LAT: runs/ct_refine/test_predictions/*_LAT_pred.npy
- Ground-truth LAT: data/test/LAT/*_LAT.npy

Outputs:
- results/metrics/metrics_all.csv
- results/summary.txt
"""

import os
import csv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

LAT_PRED_DIR = "runs/ct_refine/test_predictions"
LAT_GT_DIR = "data/test/LAT"
RESULTS_DIR = "results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    return img / (float(img.max()) + 1e-6)


def main() -> None:
    os.makedirs(METRICS_DIR, exist_ok=True)

    lat_pred_files = sorted(
        f for f in os.listdir(LAT_PRED_DIR) if f.endswith("_LAT_pred.npy")
    )
    if not lat_pred_files:
        raise RuntimeError(
            f"No LAT prediction files found in: {LAT_PRED_DIR} "
            "(expected *_LAT_pred.npy)"
        )

    rows = []
    psnr_list = []
    ssim_list = []

    for fname in lat_pred_files:
        case_id = fname.replace("_LAT_pred.npy", "")

        pred_path = os.path.join(LAT_PRED_DIR, fname)
        gt_path = os.path.join(LAT_GT_DIR, f"{case_id}_LAT.npy")

        if not os.path.exists(gt_path):
            print(f"[WARN] Missing GT LAT for {case_id} (skipping)")
            continue

        lat_pred = np.load(pred_path).astype(np.float32)
        lat_gt = np.load(gt_path).astype(np.float32)

        # Both should be (160, 128) = (y, z)
        lat_pred = normalize(lat_pred)
        lat_gt = normalize(lat_gt)

        psnr = peak_signal_noise_ratio(lat_gt, lat_pred, data_range=1.0)
        ssim = structural_similarity(lat_gt, lat_pred, data_range=1.0)

        rows.append([case_id, psnr, ssim])
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print(f"{case_id}: PSNR={psnr:.2f}, SSIM={ssim:.3f}")

    if not rows:
        raise RuntimeError("No matched GT/PRED pairs were evaluated.")

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))

    # -----------------------
    # Save CSV
    # -----------------------
    csv_path = os.path.join(METRICS_DIR, "metrics_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "PSNR", "SSIM"])
        writer.writerows(rows)

    # -----------------------
    # Save summary
    # -----------------------
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("LAT View Synthesis Evaluation (0° → 90°)\n")
        f.write("======================================\n\n")
        f.write(f"Number of evaluated test cases: {len(rows)}\n\n")
        f.write(f"Mean PSNR: {mean_psnr:.2f}\n")
        f.write(f"Mean SSIM: {mean_ssim:.3f}\n")

    print("\n===== FINAL RESULTS =====")
    print(f"Evaluated cases: {len(rows)}")
    print(f"Mean PSNR: {mean_psnr:.2f}")
    print(f"Mean SSIM: {mean_ssim:.3f}")
    print(f"\nSaved metrics CSV: {csv_path}")
    print(f"Saved summary TXT: {summary_path}")


if __name__ == "__main__":
    main()

