#!/usr/bin/env bash
# Usage: bash scripts/train_fold.sh 4

FOLD=${1:-4}

echo "Training MA3W-Net (M3A) for fold $FOLD ..."
python -m src.training.train --fold $FOLD --model m3a --cycle cycle1 --resume_mode fresh

echo "Training UNet3D for fold $FOLD ..."
python -m src.training.train --fold $FOLD --model unet3d --cycle cycle1 --resume_mode fresh

echo "Training DynUNet for fold $FOLD ..."
python -m src.training.train --fold $FOLD --model dynunet --cycle cycle1 --resume_mode fresh
