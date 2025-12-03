import os, gc, time, csv, shutil, warnings, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import argparse

from src.models.ma3w_net import M3ANeuroSeg
from src.models.unet3d_baseline import make_unet3d
from src.models.dynunet_baseline import DynUNetWrapper
from src.data.dataset_patch import BratsPatchDataset
from src.training.losses import compute_loss, dice_metric, argmax_onehot

# -----------------------------
# Helpers
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_onecycle(opt, epochs, steps_per_epoch, base_lr, last_epoch=-1):
    return OneCycleLR(
        opt,
        max_lr=base_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
        last_epoch=last_epoch
    )

def read_completed_steps_from_log(log_path):
    if not log_path.exists():
        return None
    try:
        last_updates = None
        with open(log_path, "r") as f:
            r = csv.reader(f)
            header = next(r, None)
            if not header:
                return None
            idx_upd = header.index("updates")
            for row in r:
                if not row:
                    continue
                last_updates = int(float(row[idx_upd]))
        return last_updates
    except Exception:
        return None

def load_splits(idx_dir: Path, seed: int, fold: int):
    splits_path = idx_dir / f"splits_5fold_seed{seed}.json"
    with open(splits_path, "r") as f:
        js = json.load(f)
    splits = js["splits"]
    train_ids, val_ids = [], []
    for i in range(5):
        if i == fold:
            val_ids += splits[f"fold_{i}"]
        else:
            train_ids += splits[f"fold_{i}"]
    return train_ids, val_ids

def evaluate(model, loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last_3d)
            y = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                seg_logits, bmap_logits = model(x)
            pred_oh = argmax_onehot(seg_logits)
            d = dice_metric(pred_oh, y)
            dices.append(d.cpu().numpy())
    model.train()
    if not dices:
        return 0.0, np.zeros(4, dtype=np.float32)
    dices = np.stack(dices, axis=0)
    return float(dices.mean()), dices.mean(axis=0)

def build_model(model_name: str):
    if model_name.lower() == "m3a":
        return M3ANeuroSeg(in_ch=4, num_classes=4, dims=(32,64,128,256))
    elif model_name.lower() == "unet3d":
        return make_unet3d(in_ch=4, out_ch=4, base=32)
    elif model_name.lower() in ["dynunet", "nnunet_like"]:
        return DynUNetWrapper(in_ch=4, out_ch=4, base=32)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

# -----------------------------
# Main training function
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=4, help="Fold index (0-4)")
    parser.add_argument("--model", type=str, default="m3a",
                        choices=["m3a","unet3d","dynunet","nnunet_like"])
    parser.add_argument("--cycle", type=str, default="cycle1", help="Cycle tag, e.g., cycle1")
    parser.add_argument("--resume_mode", type=str, default="fresh",
                        choices=["fresh","auto","new_cycle"])
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--boundary_w", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Config
    FOLD = args.fold
    MODEL_NAME = args.model
    CYCLE_TAG = args.cycle
    RESUME_MODE = args.resume_mode
    BASE_LR = args.base_lr
    MAX_EPOCHS = args.max_epochs
    ACCUM_STEPS = args.accum_steps
    BOUNDARY_W = args.boundary_w
    PATIENCE = args.patience
    SEED = args.seed
    PATCH = 112

    set_seed(SEED)

    project_root = Path(__file__).resolve().parents[2]
    DATA_INDEX_DIR = project_root / "data_index"
    CACHE_DIR = project_root / "cache_npy"
    RUNS_DIR = project_root / "runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    RUN_NAME = f"{MODEL_NAME}_brats2023_fold{FOLD}"
    SAVE_DIR = RUNS_DIR / f"{RUN_NAME}_{CYCLE_TAG}"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = SAVE_DIR / "log.csv"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")
    torch.backends.cudnn.benchmark = True

    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except:
            pass

    # Splits & datasets
    train_ids, val_ids = load_splits(DATA_INDEX_DIR, SEED, FOLD)
    print(f"Fold {FOLD} | Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_ds = BratsPatchDataset(train_ids, cache_dir=str(CACHE_DIR), patch=PATCH, augment=True)
    val_ds   = BratsPatchDataset(val_ids,   cache_dir=str(CACHE_DIR), patch=PATCH, augment=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=6,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4,
                              drop_last=False)

    steps_per_epoch = len(train_loader) // max(1, ACCUM_STEPS)

    # Build model + optimizer + scaler
    model = build_model(MODEL_NAME).to(device).to(memory_format=torch.channels_last_3d)
    opt   = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=5e-2)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type=="cuda"))

    last_ckpt_path = SAVE_DIR / "last.pt"
    best_ckpt_path = SAVE_DIR / "best.pt"

    start_epoch = 1
    best_val = -1.0

    # Scheduler + resume
    if RESUME_MODE == "fresh":
        sched = make_onecycle(opt, MAX_EPOCHS, steps_per_epoch, BASE_LR, last_epoch=-1)
        print("🆕 Fresh run.")
    elif RESUME_MODE == "auto" and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        try:
            opt.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
        except:
            print("⚠️ Optimizer/scaler resume failed, re-init.")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", -1.0))
        completed_steps = read_completed_steps_from_log(LOG_PATH) or max(0, (start_epoch-1)*steps_per_epoch)
        sched = make_onecycle(opt, MAX_EPOCHS, steps_per_epoch, BASE_LR, last_epoch=completed_steps-1)
        print(f"🔄 Auto-resume at epoch {start_epoch} | best={best_val:.4f}")
    else:
        # "new_cycle" or no checkpoint
        if RESUME_MODE == "new_cycle" and last_ckpt_path.exists():
            ckpt = torch.load(last_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            try:
                opt.load_state_dict(ckpt["optimizer"])
                scaler.load_state_dict(ckpt["scaler"])
            except:
                print("⚠️ Optimizer/scaler resume failed, re-init.")
            best_val = float(ckpt.get("best_val", -1.0))
            print(f"🔁 New cycle starting from {last_ckpt_path} | prev best={best_val:.4f}")
        else:
            print("🆕 Starting from scratch.")
        sched = make_onecycle(opt, MAX_EPOCHS, steps_per_epoch, BASE_LR, last_epoch=-1)

    # CSV header
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch","updates","lr","loss","dice_loss","focal_loss","bce_loss",
                 "val_mean","val_bg","val_ncr","val_ed","val_et","gpu_mem_gb"]
            )

    no_improve_epochs = 0
    update_steps = read_completed_steps_from_log(LOG_PATH) or 0

    for epoch in range(start_epoch, MAX_EPOCHS+1):
        model.train()
        running = []
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(total=len(train_loader), desc=f"{MODEL_NAME} | Fold {FOLD} | Epoch {epoch}/{MAX_EPOCHS}", ncols=120)
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device, non_blocking=True).to(memory_format=torch.channels_last_3d)
            y = batch["target"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                seg_logits, bmap_logits = model(x)
                loss, parts = compute_loss(
                    seg_logits, bmap_logits, y,
                    boundary_weight=BOUNDARY_W, gamma=2.0
                )
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step % ACCUM_STEPS == 0) or (step == len(train_loader)):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                total_planned = getattr(sched, "total_steps", MAX_EPOCHS * steps_per_epoch)
                if (sched.last_epoch + 1) < total_planned:
                    sched.step()
                update_steps += 1

            running.append((float(loss.item()*ACCUM_STEPS),
                            float(parts["dice"]), float(parts["focal"]), float(parts["bce"])))
            pbar.update(1)

        pbar.close()

        # Validation
        val_mean, val_perclass = evaluate(model, val_loader, device)
        lr_now = sched.get_last_lr()[0]
        mm = np.array(running).mean(axis=0) if running else [0,0,0,0]
        mem_gb = (torch.cuda.max_memory_allocated()/(1024**3)) if torch.cuda.is_available() else 0.0
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
        epoch_time = time.perf_counter() - t0

        print(
            f"✅ {MODEL_NAME} Fold {FOLD} | Epoch {epoch:03d}/{MAX_EPOCHS} | Updates: {update_steps:06d} | "
            f"LR: {lr_now:.2e} | Loss: {mm[0]:.4f} | DiceLoss: {mm[1]:.4f} | "
            f"Focal: {mm[2]:.4f} | BCE: {mm[3]:.4f} | "
            f"ValDice: {val_mean:.4f} | BG/NCR/ED/ET: {val_perclass.round(4)} | "
            f"Time: {epoch_time/60:.1f} min | GPU Mem: {mem_gb:.2f} GB"
        )

        # Log
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, update_steps, lr_now,
                mm[0], mm[1], mm[2], mm[3],
                val_mean, *list(val_perclass.astype(float)), mem_gb
            ])

        # Save checkpoints
        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "patch": PATCH,
            "fold": FOLD,
            "model_name": MODEL_NAME,
        }
        torch.save(ckpt, SAVE_DIR/"last.pt")

        if val_mean > best_val:
            best_val = val_mean
            torch.save(ckpt, SAVE_DIR/"best.pt")
            print(f"  ✓ Saved best checkpoint (val_mean={best_val:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"  ⚠️ No improvement for {no_improve_epochs} epochs.")

        if no_improve_epochs >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break

    print(f"🎯 Training complete. {MODEL_NAME} Fold {FOLD} | Best val dice={best_val:.4f}")

if __name__ == "__main__":
    main()
