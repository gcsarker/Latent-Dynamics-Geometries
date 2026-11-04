#!/bin/bash
#SBATCH --job-name=LDR_Training
#SBATCH --propagate=NONE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=80G
#SBATCH --output=/network/rit/dgx/dgx_dahlin_lab/Gobinda/Latent-Dynamics-Geometries/batch_training/output_%j.out
#SBATCH --error=/network/rit/dgx/dgx_dahlin_lab/Gobinda/Latent-Dynamics-Geometries/batch_training/error_%j.err

set -e

# Go to the training directory
cd /network/rit/dgx/dgx_dahlin_lab/Gobinda/Latent-Dynamics-Geometries/batch_training

# Activate Conda environment
source /network/rit/dgx/dgx_dahlin_lab/Gobinda/anaconda3/etc/profile.d/conda.sh
conda activate ldr

# Run training
python -u train_ldr.py