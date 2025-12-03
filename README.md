🧠 MA3W-Net: Multi-Attention Multi-Scale 3D Network for Brain Tumor Segmentation (BraTS-2023)

This repository contains the official implementation of MA3W-Net / M3A-NeuroSeg, a hybrid attention-powered 3D segmentation architecture developed for BraTS-2023 Glioma MRI segmentation.
It includes full preprocessing, training, inference, evaluation pipelines and baselines for fair comparison.

🔥 Highlights

Multi-Attention fusion

Local 3D ConvNeXt blocks

Windowed 3D self-attention (Swin-style without shift)

Axial/global placeholder attention for future extension

Multi-Scale decoder

Lightweight upsampling

Cross-scale attention bridges

No raw U-skip connections

Modality Gate

SE-style gating for the 4 MRI modalities

Boundary-Aware Supervision

Boundary head + BCE loss on tumor edges

5-Fold Training

Stratified by tumor size tertiles

Ensemble Support

MA3W-Net

3D U-Net (MONAI)

DynUNet (nnU-Net-like)

Tri-ensemble (average logits)

Sliding-Window Inference

Full-volume predictions

Optional test-time augmentation

📦 Repository Structure
## 📦 Repository Structure

```
MA3W-Net/
│── src/
│   ├── data/
│   │   ├── index_splits.py
│   │   ├── preprocess_npy.py
│   │   └── dataset_patch.py
│   ├── models/
│   │   ├── m3a_neuroseg.py
│   │   ├── unet3d_monai.py
│   │   └── dynunet_wrapper.py
│   ├── training/
│   │   ├── train_5fold.py
│   │   └── train_single_cycle.py
│   ├── inference/
│   │   ├── infer_sliding.py
│   │   └── visualize_case.py
│   └── eval/
│       └── eval_folds.py
│
│── scripts/
│   ├── run_training.sh
│   ├── run_evaluation.sh
│   └── setup_env.sh
│
│── cache_npy/             # Preprocessed volume cache (generated)
│── runs/                  # Trained checkpoints + logs (generated)
│── out_eval/              # Evaluation outputs (generated)
│── data_index/            # Dataset index + splits (generated)
│── requirements.txt
│── README.md
```


📥 Dataset: BraTS-2023 (GLI)

Download (Kaggle mirror):
🔗 https://www.kaggle.com/datasets/bhavesh907/bra-ts-2023-dataset

Structure:

BraTS2023/
   ├── brats2023/
       ├── BraTS-GLI-00000-000/
           ├── BraTS-GLI-00000-000-t1c.nii.gz
           ├── BraTS-GLI-00000-000-t1n.nii.gz
           ├── BraTS-GLI-00000-000-t2w.nii.gz
           ├── BraTS-GLI-00000-000-t2f.nii.gz
           ├── BraTS-GLI-00000-000-seg.nii.gz

⚙️ Installation
conda create -n ma3w python=3.10 -y
conda activate ma3w
pip install -r requirements.txt


Or manually:

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel simpleitk pandas scikit-image matplotlib tqdm einops

🧭 Complete Pipeline to Reproduce Results

Below are the exact steps used in our experiments.

1️⃣ Create Index & 5-Fold Splits
python src/data/index_splits.py \
    --root /path/to/BraTS2023/brats2023 \
    --out data_index/


Creates:

brats2023_index.csv
splits_5fold_seed42.json


Splits are stratified by tumor size tertiles.

2️⃣ Preprocess Into .npy Cache (Fast I/O)
python src/data/preprocess_npy.py \
    --index data_index/brats2023_index.csv \
    --cache cache_npy/


This script performs:

Resample → 1 mm³

Z-score normalization

Tight bounding box around brain/tumor

Pad to ≥128³

Save modalities as .npy

Save meta information

3️⃣ Train MA3W-Net (Fold Example: 4)
python src/training/train_single_cycle.py \
    --model m3a \
    --fold 4 \
    --cycle 1 \
    --epochs 80 \
    --patch 112


Or using the convenience script:

bash scripts/run_training.sh m3a fold=4 cycle=1


Outputs:

runs/m3a_brats2023_fold4_cycle1/best.pt
runs/m3a_brats2023_fold4_cycle1/log.csv

4️⃣ Train Baselines
3D U-Net
bash scripts/run_training.sh unet3d fold=4 cycle=1

DynUNet (nnU-Net-like)
bash scripts/run_training.sh dynunet fold=4 cycle=1

5️⃣ Full-Volume Inference
python src/inference/infer_sliding.py \
    --case BraTS-GLI-00000-000 \
    --model m3a \
    --fold 4

6️⃣ Visualization
python src/inference/visualize_case.py \
    --case BraTS-GLI-00000-000 \
    --slice 80 \
    --fold 4


Displays:

4 MRI channels

Ground truth

UNet3D, DynUNet, MA3W-Net

Ensemble

7️⃣ Evaluate All Methods Across All Folds
bash scripts/run_evaluation.sh


Outputs:

out_eval/
   ├── MA3WNet/perfold_summary.csv
   ├── UNet3D/perfold_summary.csv
   ├── DynUNet/perfold_summary.csv
   ├── TriEnsemble/perfold_summary.csv
   ├── all_methods_perfold_summary.csv


Metrics:

Dice (NCR, ED, ET)

HD95

ET, TC, WT composites

🎯 Pretrained Weights

All trained model weights are available here:

🔗 Google Drive Checkpoints
https://drive.google.com/drive/folders/YOUR_FOLDER_ID

Contains:

MA3W-Net best model (fold-4)

3D U-Net best model

DynUNet best model

Optional tri-ensemble configurations

Place them into:

runs/<model>_brats2023_fold4_cycle1/best.pt

📊 Example MA3W-Net Performance (Fold: 4)
Class	Dice	HD95
NCR/NET	0.XX	XX.X
ED	0.XX	XX.X
ET	0.XX	XX.X
Mean	0.XX	XX.X

(Replace with your actual computed results.)

🧠 MA3W-Net Architecture

Key components:

ConvNeXt-3D local pathway

Windowed 3D attention blocks

Modality gate

Cross-scale decoder

Boundary head

Tri-model ensemble option

(Architecture diagram can be added later if needed.)

🧪 Citation

If you use this repository in your research:

@article{manzoor2025ma3w,
  title={MA3W-Net: Multi-Attention Multi-Scale 3D Network for Brain Tumor Segmentation},
  author={Mohammad, Manzoor and Vijaya Babu, Burra},
  year={2025}
}

❤️ Acknowledgements

BraTS Challenge Dataset

MONAI & PyTorch teams

SimpleITK tooling

Community MRI research contributors
