MA3W-Net: Multi-Attention Multi-Scale 3D Network for Brain Tumor Segmentation (BraTS-2023)

This repository contains the official implementation of MA3W-Net / M3A-NeuroSeg, a hybrid attention-powered 3D segmentation architecture developed for BraTS-2023 Glioma MRI segmentation.
It includes full preprocessing, training, inference, evaluation pipelines and baselines for fair comparison.

Highlights:

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
🔗 (https://www.synapse.org/Synapse:syn51156910)

Structure:
```
BraTS2023/
   ├── brats2023/
       ├── BraTS-GLI-00000-000/
           ├── BraTS-GLI-00000-000-t1c.nii.gz
           ├── BraTS-GLI-00000-000-t1n.nii.gz
           ├── BraTS-GLI-00000-000-t2w.nii.gz
           ├── BraTS-GLI-00000-000-t2f.nii.gz
           ├── BraTS-GLI-00000-000-seg.nii.gz
```
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
```
python src/data/preprocess_npy.py \
    --index data_index/brats2023_index.csv \
    --cache cache_npy/
```

This script performs:

Resample → 1 mm³

Z-score normalization

Tight bounding box around brain/tumor

Pad to ≥128³

Save modalities as .npy

Save meta information

3️⃣ Train MA3W-Net (Fold Example: 4)
```
python src/training/train_single_cycle.py \
    --model m3a \
    --fold 4 \
    --cycle 1 \
    --epochs 80 \
    --patch 112

```
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
```
python src/inference/infer_sliding.py \
    --case BraTS-GLI-00000-000 \
    --model m3a \
    --fold 4
```
6️⃣ Visualization
```
python src/inference/visualize_case.py \
    --case BraTS-GLI-00000-000 \
    --slice 80 \
    --fold 4
```

Displays:

4 MRI channels

Ground truth

UNet3D, DynUNet, MA3W-Net

Ensemble

7️⃣ Evaluate All Methods Across All Folds
bash scripts/run_evaluation.sh


Outputs:
```
out_eval/
   ├── MA3WNet/perfold_summary.csv
   ├── UNet3D/perfold_summary.csv
   ├── DynUNet/perfold_summary.csv
   ├── TriEnsemble/perfold_summary.csv
   ├── all_methods_perfold_summary.csv

```
Metrics:

Dice (NCR, ED, ET)

HD95

ET, TC, WT composites

🎯 Pretrained Weights

All trained model weights are available here:

🔗 Google Drive Checkpoints
#WILL BE SHARED ON REQUEST write to- manzoor.rsrch@gmail.com### Model Checkpoints

The trained model checkpoints for MA3W-Net, 3D U-Net, DynUNet, and the tri-ensemble setting are large and are therefore not included directly in this repository.

Researchers who wish to access the pretrained weights for verification, benchmarking, or further experimentation may request them by contacting the corresponding author at:

**Email:** manzoor.rsrch@gmail.com

The full training and inference code, along with dataset preparation scripts, is provided in this repository to support complete reproducibility of the experiments reported in the manuscript.

Contains:

MA3W-Net best model (fold-4)

3D U-Net best model

DynUNet best model

Optional tri-ensemble configurations

Place them into:

runs/<model>_brats2023_fold4_cycle1/best.pt

📊  MA3W-Net Performance (on Fold: 4 and 5-fold mean on ensemble)

```
Method	                     Dice Mean	      Dice ET	   Dice TC	   Dice WT	   HD95 Mean
UNet3D	                     0.8189 ± 0.0124	0.8596	   0.9031	   0.9250	   5.1464
DynUNet	                     0.8527 ± 0.0088	0.8786	   0.9318	   0.9434	   3.8234
MA3W-Net	                     0.8389 ± 0.0101	0.8735	   0.9166	   0.9341	   4.3774
MA3W-Net+TTA	               0.845±0.01	      0.880±0.005	0.922±0.006	0.937±0.005	4.10±0.15
Conv-Ensemble (UNet3+DynUNet)	0.860±0.008	      0.885±0.01	0.933±0.009	0.947±0.005	3.8±0.3
Tri-Ensemble 
(UNet3D+DynUNet+MA3W-Net)	   0.870±0.008	      0.892±0.01	0.938±0.009	0.952±0.005	3.6±0.3
Tri-Ensemble+ TTA	            0.875±0.01	      0.900±0.008	0.944±0.006	0.955±0.005	3.4–±0.3
Tri-Ensemble 5Fold Mean      	0.8760            0.9019	   0.9379	   0.9456		3.64
(U-Net + DynUNet + MA3W-Net)
```
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
