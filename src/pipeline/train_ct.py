import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipeline.dataset import VbpToCTDataset
from src.pipeline.unet3d import UNet3D


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, path)


@torch.no_grad()
def save_debug_slices(out_dir, case_id, vbp, ct_gt, ct_pred, epoch):
    """
    Save a quick 3-slice comparison as a PNG:
      Vbp vs CT_gt vs CT_pred at z = [0, mid, last]

    vbp, ct_gt, ct_pred expected shapes: (1, D, H, W)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    # Convert tensors -> numpy
    v = vbp.squeeze(0).detach().cpu().numpy()      # (D,H,W)
    g = ct_gt.squeeze(0).detach().cpu().numpy()
    p = ct_pred.squeeze(0).detach().cpu().numpy()

    D = v.shape[0]
    zs = [0, D // 2, D - 1]

    fig = plt.figure(figsize=(12, 6))
    for i, z in enumerate(zs):
        # Row 1: Vbp
        ax = fig.add_subplot(3, 3, 1 + i)
        ax.imshow(v[z], cmap="gray")
        ax.set_title(f"Vbp z={z}")
        ax.axis("off")

        # Row 2: GT
        ax = fig.add_subplot(3, 3, 4 + i)
        ax.imshow(g[z], cmap="gray")
        ax.set_title(f"CT_gt z={z}")
        ax.axis("off")

        # Row 3: Pred
        ax = fig.add_subplot(3, 3, 7 + i)
        ax.imshow(p[z], cmap="gray")
        ax.set_title(f"CT_pred z={z}")
        ax.axis("off")

    fig.suptitle(f"{case_id} | epoch {epoch}", y=0.98)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"epoch_{epoch:03d}_{case_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def train_one_epoch(model, loader, optimizer, loss_fn, device, log_every=20):
    model.train()
    running = 0.0
    n = 0

    for step, batch in enumerate(loader):
        vbp = batch["vbp"].to(device)  # (B,1,D,H,W)
        ct = batch["ct"].to(device)    # (B,1,D,H,W)

        optimizer.zero_grad(set_to_none=True)
        pred = model(vbp)
        loss = loss_fn(pred, ct)
        loss.backward()
        optimizer.step()

        running += loss.item() * vbp.size(0)
        n += vbp.size(0)

        if (step + 1) % log_every == 0:
            print(f"  step {step+1:04d}/{len(loader):04d} | loss {loss.item():.6f}")

    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    n = 0

    for batch in loader:
        vbp = batch["vbp"].to(device)
        ct = batch["ct"].to(device)
        pred = model(vbp)
        loss = loss_fn(pred, ct)

        running += loss.item() * vbp.size(0)
        n += vbp.size(0)

    return running / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="Project root (xrayct_project)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0, help="Use 0 on macOS for stability")
    parser.add_argument("--save_dir", type=str, default="runs/ct_refine")
    parser.add_argument("--save_debug_every", type=int, default=1)
    args = parser.parse_args()

    device = get_device()
    print("Device:", device)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    dbg_dir = os.path.join(args.save_dir, "debug_slices")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(dbg_dir, exist_ok=True)

    # Datasets
    train_ds = VbpToCTDataset(root_dir=args.root, split="train")
    val_ds = VbpToCTDataset(root_dir=args.root, split="val")

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Model
    model = UNet3D(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)

    # Loss + optimizer
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    print("\nStarting training (CT loss only)")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        dt = time.time() - t0
        print(f"\nEpoch {epoch:03d}/{args.epochs:03d} | train {train_loss:.6f} | val {val_loss:.6f} | {dt:.1f}s")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(best_path, model, optimizer, epoch, best_val)
            print(f"  Saved best checkpoint: {best_path} (val {best_val:.6f})")

        # Save periodic checkpoint
        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(last_path, model, optimizer, epoch, best_val)

        # Save debug slices from first val sample
        if args.save_debug_every > 0 and (epoch % args.save_debug_every == 0):
            batch = next(iter(val_loader))
            vbp = batch["vbp"].to(device)
            ct = batch["ct"].to(device)
            case_id = batch["id"][0] if isinstance(batch["id"], list) else batch["id"]

            pred = model(vbp)

            # save only first sample in batch
            save_debug_slices(
                out_dir=dbg_dir,
                case_id=case_id,
                vbp=vbp[0].detach().cpu(),
                ct_gt=ct[0].detach().cpu(),
                ct_pred=pred[0].detach().cpu(),
                epoch=epoch,
            )
            print(f"  Saved debug slices in: {dbg_dir}")

    print("\nTraining finished.")
    print("Best val loss:", best_val)
    print("Checkpoints:", ckpt_dir)
    print("Debug images:", dbg_dir)


if __name__ == "__main__":
    main()

