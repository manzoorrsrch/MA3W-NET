cat > scripts/visualize_case.py << 'EOF'
#!/usr/bin/env python3
"""
Visualize UNet3D, DynUNet, MA3W-Net, and Ensemble predictions for a single case.
Usage:
    python scripts/visualize_case.py --case BraTS-GLI-00000-000 --fold 4
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.ma3w_net import M3ANeuroSeg
from src.models.unet3d_baseline import make_unet3d
from src.models.dynunet_baseline import DynUNetWrapper
from src.inference.infer_sliding import sw_logits
from src.eval.metrics import composites_from_lbl

# -----------------------------
# Helpers
# -----------------------------

def load_case(cache_dir: Path, case_id: str):
    d = cache_dir / case_id
    vol = np.stack([
        np.load(d/"t1c.npy"),
        np.load(d/"t1n.npy"),
        np.load(d/"t2w.npy"),
        np.load(d/"t2f.npy"),
    ], axis=0)
    seg = np.load(d/"seg.npy").astype(np.int64)
    return vol, seg

def load_model(ckpt_path, builder, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = builder().to(device).to(memory_format=torch.channels_last_3d)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

# -----------------------------
# Visualization
# -----------------------------

def visualize(vol, gt_lbl, pred_unet, pred_dyn, pred_m3a, pred_ens, slice_idx):
    comps_gt  = composites_from_lbl(gt_lbl)
    comps_u   = composites_from_lbl(pred_unet)
    comps_d   = composites_from_lbl(pred_dyn)
    comps_m   = composites_from_lbl(pred_m3a)
    comps_e   = composites_from_lbl(pred_ens)

    titles = ["MRI T1c", "MRI T1n", "MRI T2w", "MRI FLAIR"]
    nrows, ncols = 6, 4
    plt.figure(figsize=(18, 4*nrows))

    row = 1

    # ---------------- Row 1: MRI ----------------
    for i in range(4):
        plt.subplot(nrows, ncols, (row-1)*ncols + i + 1)
        plt.imshow(vol[i, slice_idx], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    row += 1

    # ---------------- Row 2: Ground Truth ----------------
    gt_shows = [gt_lbl, comps_gt["ET"], comps_gt["TC"], comps_gt["WT"]]
    gt_titles = ["GT Labels", "GT ET", "GT TC", "GT WT"]

    for i in range(4):
        plt.subplot(nrows, ncols, (row-1)*ncols + i + 1)
        plt.imshow(gt_shows[i][slice_idx], cmap="viridis")
        plt.title(gt_titles[i])
        plt.axis("off")
    row += 1

    # ---------------- Helper ----------------
    def show_pred_row(pred_lbl, comps, prefix):
        nonlocal row
        imgs = [pred_lbl, comps["ET"], comps["TC"], comps["WT"]]
        labels = [f"{prefix} Labelmap", f"{prefix} ET", f"{prefix} TC", f"{prefix} WT"]
        for i in range(4):
            plt.subplot(nrows, ncols, (row-1)*ncols + i + 1)
            plt.imshow(imgs[i][slice_idx], cmap="viridis")
            plt.title(labels[i])
            plt.axis("off")
        row += 1

    # ---------------- Rows 3–6 ----------------
    show_pred_row(pred_unet, comps_u, "UNet3D")
    show_pred_row(pred_dyn,  comps_d,  "DynUNet")
    show_pred_row(pred_m3a, comps_m,  "MA3WNet")
    show_pred_row(pred_ens, comps_e,  "Ensemble")

    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, required=True, help="Case ID under cache_npy/")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--slice", type=int, default=None, help="Axial slice index")
    args = ap.parse_args()

    # Paths
    root = Path(__file__).resolve().parents[1]
    cache_dir = root / "cache_npy"
    runs = root / "runs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Load case ----------------
    vol, gt = load_case(cache_dir, args.case)
    slice_idx = args.slice or vol.shape[1] // 2

    # ---------------- Load all 3 models ----------------
    ckpt_unet = runs / f"unet3d_brats2023_fold{args.fold}_cycle1" / "best.pt"
    ckpt_dyn  = runs / f"dynunet_brats2023_fold{args.fold}_cycle1" / "best.pt"
    ckpt_m3a  = runs / f"m3a_brats2023_fold{args.fold}_cycle1" / "best.pt"

    m_unet = load_model(ckpt_unet, make_unet3d, device)
    m_dyn  = load_model(ckpt_dyn,  DynUNetWrapper, device)
    m_m3a  = load_model(ckpt_m3a,  lambda: M3ANeuroSeg(in_ch=4, num_classes=4, dims=(32,64,128,256)), device)

    # ---------------- Inference ----------------
    PATCH = 112
    OVERLAP = 0.5

    L_unet = sw_logits(m_unet, vol, roi=PATCH, overlap=OVERLAP)
    L_dyn  = sw_logits(m_dyn,  vol, roi=PATCH, overlap=OVERLAP)
    L_m3a  = sw_logits(m_m3a,  vol, roi=PATCH, overlap=OVERLAP)

    pred_unet = L_unet.argmax(0)
    pred_dyn  = L_dyn.argmax(0)
    pred_m3a  = L_m3a.argmax(0)

    pred_ens  = ((L_unet + L_dyn + L_m3a) / 3).argmax(0)

    # ---------------- Visualize ----------------
    visualize(vol, gt, pred_unet, pred_dyn, pred_m3a, pred_ens, slice_idx)

if __name__ == "__main__":
    main()
