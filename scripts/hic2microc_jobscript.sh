#!/bin/bash
#SBATCH --job-name=impute_hic2microc_full
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-1:00
#SBATCH --mem=32G
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drakedu@college.harvard.edu

# Load Python.
module load python

# Activate the environment. mamba create -n impute_hic2microc \ python=3.10 \ pytorch torchvision torchaudio pytorch-cuda=11.8 \ cooler numpy pandas scipy einops \
conda activate impute_hic2microc

# Run HiC2MicroC imputation.
python HiC2MicroC/src/HiC2MicroC.py \
  -f1 data/GM12878.GSE115524.Homo_Sapiens.CTCF.b1.chr21.mcool::resolutions/5000 \
  -f2 data/hg38.chr21.sizes \
  -f3 impute_hic2microc_full