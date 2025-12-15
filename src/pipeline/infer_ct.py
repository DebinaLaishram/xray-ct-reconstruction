import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.pipeline.dataset import VbpToCTDataset
from src.pipeline.unet3d import UNet3D


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_debug_slice(out_path, case_id, vbp, ct_gt, ct_pred):
    """
    vbp, ct_gt, ct_pred: torch tensors of shape (1, D, H, W)
    """
    v = vbp.squeeze(0).cpu().numpy()      # (D,H,W)
    g = ct_gt.squeeze(0).cpu().numpy()
    p = ct_pred.squeeze(0).cpu().numpy()

    D = v.shape[0]
    zs = [0, D // 2, D - 1]

    fig = plt.figure(figsize=(12, 6))

    for i, z in enumerate(zs):
        ax = fig.add_subplot(3, 3, 1 + i)
        ax.imshow(v[z], cmap="gray")
        ax.set_title(f"Vbp z={z}")
        ax.axis("off")

        ax = fig.add_subplot(3, 3, 4 + i)
        ax.imshow(g[z], cmap="gray")
        ax.set_title(f"CT_gt z={z}")
        ax.axis("off")

        ax = fig.add_subplot(3, 3, 7 + i)
        ax.imshow(p[z], cmap="gray")
        ax.set_title(f"CT_pred z={z}")
        ax.axis("off")

    fig.suptitle(case_id)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    device = get_device()
    print(f"Device: {device}")

    checkpoint_path = "runs/ct_refine/checkpoints/baseline_ct_only.pt"
    output_dir = "runs/ct_refine/test_predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    test_ds = VbpToCTDataset(root_dir=".", split="test")

    # Model
    model = UNet3D(in_channels=1, out_channels=1, base_channels=16)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print("Loaded model checkpoint:", checkpoint_path)

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Infer CT (test)"):
            sample = test_ds[idx]

            vbp = sample["vbp"].to(device)   # (1,D,H,W)
            ct_gt = sample["ct"]              # (1,D,H,W)
            case_id = sample["id"]

            ct_pred = model(vbp.unsqueeze(0)) # (1,1,D,H,W)
            ct_pred = ct_pred.squeeze(0)      # (1,D,H,W)

            np.save(
                os.path.join(output_dir, f"{case_id}_CT_pred.npy"),
                ct_pred.cpu().numpy().astype(np.float32)
            )

            if idx < 10:
                save_debug_slice(
                    out_path=os.path.join(output_dir, f"{case_id}_debug.png"),
                    case_id=case_id,
                    vbp=vbp,
                    ct_gt=ct_gt,
                    ct_pred=ct_pred,
                )

    print("Inference complete.")
    print("Saved CT predictions to:", output_dir)


if __name__ == "__main__":
    main()
