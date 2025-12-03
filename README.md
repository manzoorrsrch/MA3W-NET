# MA3W-Net: Multi-Path 3D Attention-Aware Windowed Network for Brain Tumor Segmentation (BraTS 2023)

This repository contains the full implementation of **MA3W-Net**, a lightweight but high-accuracy
3D brain tumor segmentation model evaluated on the **BraTS 2023** dataset.  
It includes:

- ? End-to-end preprocessing (SimpleITK ? cropped, resampled, z-score normalized, padded NPZ cache)  
- ? Tumor-aware 3D patch sampling  
- ? MA3W-Net architecture with multi-path attention and boundary-map branch  
- ? Training pipeline (One-Cycle LR, AMP, resume cycles, GDrive sync, checkpoints)  
- ? Baseline models: **3D UNet**, **DynUNet**  
- ? Full cross-validation evaluation for all folds  
- ? Single-case inference and visualization tools  
- ? Tri-ensemble evaluation  
- ? Publication-ready metric tables (Dice, HD95, ET/TC/WT composites)

---

# ? Model Summary

### **MA3W-Net includes:**
- Multi-path convolutional + attention downsampling  
- Boundary prediction branch for sharper tumor borders  
- Lightweight decoder  
- Stable normalization and residual units  
- Designed for **112×112×112** patch-based segmentation  
- Trained using soft-Dice + focal + boundary loss

### **Key Results (Fold-4 Example)**
| Method | Dice Mean | WT | TC | ET |
|-------|-----------|----|----|----|
| UNet3D | ~0.86 | - | - | - |
| DynUNet | ~0.88 | - | - | - |
| **MA3W-Net** | **~0.90** | **?** | **?** | **?** |
| **Tri-Model Ensemble** | **~0.915** | SOTA-like | TC/ET boosted | stable |

---

# ?? Installation

```bash
pip install -r requirements.txt