"""
Generate qualitative example pairs for LAT view synthesis.

For each test case:
AP input -> Predicted LAT -> Ground-truth LAT

Outputs:
results/examples/LIDC-IDRI-XXXX/
  - AP.png
  - LAT_pred.png
  - LAT_gt.png
  - metrics.txt
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

AP_DIR = "data/test/AP"
LAT_GT_DIR = "data/test/LAT"
LAT_PRED_DIR = "runs/ct_refine/test_predictions"

RESULTS_DIR = "results/examples"
METRICS_CSV = "results/metrics/metrics_all.csv"


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    return (img - img.min()) / (img.max() - img.min() + 1e-6)


def save_png(path: str, img: np.ndarray) -> None:
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def load_metrics(csv_path: str):
    metrics = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["case_id"]] = {
                "PSNR": float(row["PSNR"]),
                "SSIM": float(row["SSIM"]),
            }
    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics = load_metrics(METRICS_CSV)

    lat_pred_files = sorted(
        f for f in os.listdir(LAT_PRED_DIR) if f.endswith("_LAT_pred.npy")
    )

    print(f"Generating examples for {len(lat_pred_files)} test cases...\n")

    for fname in lat_pred_files:
        case_id = fname.replace("_LAT_pred.npy", "")

        ap_path = os.path.join(AP_DIR, f"{case_id}_AP.npy")
        lat_pred_path = os.path.join(LAT_PRED_DIR, fname)
        lat_gt_path = os.path.join(LAT_GT_DIR, f"{case_id}_LAT.npy")

        if not (os.path.exists(ap_path) and os.path.exists(lat_gt_path)):
            print(f"[WARN] Missing data for {case_id}, skipping.")
            continue

        # Load data
        ap = np.load(ap_path)
        lat_pred = np.load(lat_pred_path)
        lat_gt = np.load(lat_gt_path)

        # Normalize for visualization
        ap = normalize(ap)
        lat_pred = normalize(lat_pred)
        lat_gt = normalize(lat_gt)

        # Create output directory
        case_dir = os.path.join(RESULTS_DIR, case_id)
        os.makedirs(case_dir, exist_ok=True)

        # Save images
        save_png(os.path.join(case_dir, "AP.png"), ap)
        save_png(os.path.join(case_dir, "LAT_pred.png"), lat_pred)
        save_png(os.path.join(case_dir, "LAT_gt.png"), lat_gt)

        # Save metrics
        m = metrics.get(case_id, None)
        metrics_path = os.path.join(case_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            if m is not None:
                f.write(f"PSNR: {m['PSNR']:.2f}\n")
                f.write(f"SSIM: {m['SSIM']:.3f}\n")
            else:
                f.write("Metrics not available\n")

        print(f"Saved example for {case_id}")

    print("\n✅ Example generation complete.")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

