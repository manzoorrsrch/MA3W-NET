import os, gc, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from src.models.ma3w_net import M3ANeuroSeg
from src.models.unet3d_baseline import make_unet3d
from src.models.dynunet_baseline import DynUNetWrapper
from src.inference.infer_sliding import sw_logits
from src.eval.metrics import one_hot_labels, compute_composites, safe_comp_metrics

# -----------------------------
# Config helpers
# -----------------------------

def get_project_paths():
    root = Path(__file__).resolve().parents[2]
    return dict(
        root=root,
        cache=root/"cache_npy",
        idx=root/"data_index",
        runs=root/"runs",
        out=root/"evaluation_results"
    )

def get_val_ids_for_fold(idx_dir: Path, seed: int, fold: int):
    splits_path = idx_dir / f"splits_5fold_seed{seed}.json"
    with open(splits_path,"r") as f:
        js = json.load(f)
    splits = js["splits"]
    val_ids = splits[f"fold_{fold}"]
    return val_ids

def build_m3a():
    return M3ANeuroSeg(in_ch=4, num_classes=4, dims=(32,64,128,256))

def build_unet3d():
    return make_unet3d(in_ch=4, out_ch=4, base=32)

def build_dynunet():
    return DynUNetWrapper(in_ch=4, out_ch=4, base=32)

# -----------------------------
# Load NPZ case
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

# -----------------------------
# Method definitions
# -----------------------------

def make_methods(runs_dir: Path):
    """
    Creates method config list.
    Checkpoint templates use fold placeholder {fold}.
    """
    return [
        {
            "name":  "UNet3D",
            "label": "3D U-Net",
            "type":  "single",
            "ckpt_tmpl": str(runs_dir / "unet3d_brats2023_fold{fold}_cycle1" / "best.pt"),
            "builder": build_unet3d,
        },
        {
            "name":  "DynUNet",
            "label": "DynUNet",
            "type":  "single",
            "ckpt_tmpl": str(runs_dir / "dynunet_brats2023_fold{fold}_cycle1" / "best.pt"),
            "builder": build_dynunet,
        },
        {
            "name":  "MA3WNet",
            "label": "MA3W-Net",
            "type":  "single",
            "ckpt_tmpl": str(runs_dir / "m3a_brats2023_fold{fold}_cycle1" / "best.pt"),
            "builder": build_m3a,
        },
        {
            "name":  "TriEnsemble",
            "label": "3D U-Net + DynUNet + MA3W-Net",
            "type":  "ensemble",
            "ckpt_tmpls": [
                str(runs_dir / "unet3d_brats2023_fold{fold}_cycle1" / "best.pt"),
                str(runs_dir / "dynunet_brats2023_fold{fold}_cycle1" / "best.pt"),
                str(runs_dir / "m3a_brats2023_fold{fold}_cycle1" / "best.pt"),
            ],
            "builders": [build_unet3d, build_dynunet, build_m3a],
        },
    ]

# -----------------------------
# Model loading for each method/fold
# -----------------------------

def load_models_for_fold(method_cfg, fold, device):
    if method_cfg["type"] == "single":
        ckpt_path = method_cfg["ckpt_tmpl"].format(fold=fold)
        assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=device)
        model = method_cfg["builder"]().to(device).to(memory_format=torch.channels_last_3d)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return [model]

    elif method_cfg["type"] == "ensemble":
        models = []
        for tmpl, builder in zip(method_cfg["ckpt_tmpls"], method_cfg["builders"]):
            ckpt_path = tmpl.format(fold=fold)
            assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
            ckpt = torch.load(ckpt_path, map_location=device)
            m = builder().to(device).to(memory_format=torch.channels_last_3d)
            m.load_state_dict(ckpt["model"])
            m.eval()
            models.append(m)
        return models

    else:
        raise ValueError(f"Unknown method type: {method_cfg['type']}")

# -----------------------------
# Inference per method/fold
# -----------------------------

def infer_logits(models, vol, roi=112, overlap=0.5, tta=False):
    # Ensemble average of logits
    logits_sum = None
    for m in models:
        L = sw_logits(m, vol, roi=roi, overlap=overlap)
        logits_sum = L if logits_sum is None else logits_sum + L
    logits = logits_sum / len(models)

    if tta:
        vol_f = vol[:, ::-1]
        logits_sum = None
        for m in models:
            Lf = sw_logits(m, vol_f, roi=roi, overlap=overlap)[:, ::-1]
            logits_sum = Lf if logits_sum is None else logits_sum + Lf
        logits_tta = logits_sum / len(models)
        logits = 0.5 * logits + 0.5 * logits_tta

    return logits

# -----------------------------
# Per-fold evaluation
# -----------------------------

def evaluate_method_on_fold(method_cfg, fold, paths, tta=False, save_case_csv=True, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = paths["cache"]
    idx_dir   = paths["idx"]
    out_root  = paths["out"]

    models = load_models_for_fold(method_cfg, fold, device)
    val_ids = get_val_ids_for_fold(idx_dir, seed, fold)

    # MONAI metrics (per-class NCR/ED/ET, no background)
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
    dice_metric.reset(); hd95_metric.reset()

    case_rows = []
    PATCH = 112
    OVERLAP = 0.5
    NUM_CLASSES = 4

    for cid in tqdm(val_ids, desc=f"{method_cfg['label']} | Fold {fold}", ncols=110):
        vol, seg = load_case(cache_dir, cid)

        logits = infer_logits(models, vol, roi=PATCH, overlap=OVERLAP, tta=tta)

        # hard labels
        pred_lbl = torch.from_numpy(logits).argmax(0, keepdim=True).unsqueeze(0)  # (1,1,Z,Y,X)
        gt_lbl   = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).to(pred_lbl.device)

        # one-hot
        pred_oh = one_hot_labels(pred_lbl, C=NUM_CLASSES)
        gt_oh   = one_hot_labels(gt_lbl,   C=NUM_CLASSES)

        # add to per-class metrics (NCR,ED,ET)
        dice_metric(y_pred=pred_oh, y=gt_oh)
        hd95_metric(y_pred=pred_oh, y=gt_oh)

        # composites
        comps = compute_composites(pred_oh.bool(), gt_oh.bool())
        dET, hET = safe_comp_metrics(*comps["ET"])
        dTC, hTC = safe_comp_metrics(*comps["TC"])
        dWT, hWT = safe_comp_metrics(*comps["WT"])

        case_rows.append(dict(
            case_id=cid,
            dice_ET=dET, dice_TC=dTC, dice_WT=dWT,
            hd95_ET=hET, hd95_TC=hTC, hd95_WT=hWT
        ))

        del logits, pred_lbl, gt_lbl, pred_oh, gt_oh
        gc.collect()

    dice_arr = dice_metric.aggregate().cpu().numpy()  # (Ncases,3)
    hd95_arr = hd95_metric.aggregate().cpu().numpy()  # (Ncases,3)
    dice_arr = np.nan_to_num(dice_arr, nan=0.0)
    hd95_arr = np.nan_to_num(hd95_arr, nan=0.0)

    dice_cls = dice_arr.mean(axis=0)
    hd95_cls = hd95_arr.mean(axis=0)

    dice_mean = float(dice_cls.mean())
    hd95_mean = float(hd95_cls.mean())

    df_cases = pd.DataFrame(case_rows)

    def nm(col):
        return float(df_cases[col].dropna().mean()) if col in df_cases.columns else float("nan")

    summary = dict(
        method    = method_cfg["name"],
        label     = method_cfg["label"],
        fold      = fold,
        dice_mean = dice_mean,
        dice_ncr  = float(dice_cls[0]),
        dice_ed   = float(dice_cls[1]),
        dice_et   = float(dice_cls[2]),
        hd95_mean = hd95_mean,
        hd95_ncr  = float(hd95_cls[0]),
        hd95_ed   = float(hd95_cls[1]),
        hd95_et   = float(hd95_cls[2]),
        dice_ET   = nm("dice_ET"),
        dice_TC   = nm("dice_TC"),
        dice_WT   = nm("dice_WT"),
        hd95_ET   = nm("hd95_ET"),
        hd95_TC   = nm("hd95_TC"),
        hd95_WT   = nm("hd95_WT"),
    )

    if save_case_csv:
        out_dir = out_root / method_cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        df_cases.to_csv(out_dir / f"fold{fold}_cases.csv", index=False)

    return summary, df_cases

# -----------------------------
# Main: evaluate all methods across folds
# -----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0,1,2,3,4",
                    help="Comma-separated folds, e.g. 0,1,2,3,4")
    ap.add_argument("--tta", action="store_true", help="Use simple Z-flip TTA")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    folds = [int(f.strip()) for f in args.folds.split(",") if f.strip()!=""]
    tta = args.tta
    seed = args.seed

    paths = get_project_paths()
    paths["out"].mkdir(parents=True, exist_ok=True)
    methods = make_methods(paths["runs"])

    all_summaries = []

    for method_cfg in methods:
        method_rows = []
        for f in folds:
            s, df_cases = evaluate_method_on_fold(
                method_cfg, f, paths, tta=tta, save_case_csv=True, seed=seed
            )
            method_rows.append(s)

        df_method = pd.DataFrame(method_rows)
        out_dir   = paths["out"] / method_cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        df_method.to_csv(out_dir / "perfold_summary.csv", index=False)
        all_summaries.append(df_method)

    df_all = pd.concat(all_summaries, ignore_index=True)
    df_all.to_csv(paths["out"] / "all_methods_perfold_summary.csv", index=False)
    print("\n✅ Saved per-fold summary for all methods →", paths["out"] / "all_methods_perfold_summary.csv")

if __name__ == "__main__":
    main()
