#!/usr/bin/env bash
# Usage: bash scripts/evaluate_all.sh

python -m src.eval.evaluate_folds --folds 0,1,2,3,4
