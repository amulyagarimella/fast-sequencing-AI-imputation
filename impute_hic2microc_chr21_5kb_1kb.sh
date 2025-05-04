#!/bin/bash
#SBATCH --job-name=impute_hic2microc_chr21_5kb_1kb
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-0:30
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drakedu@college.harvard.edu

# Load Python.
module load python

# Activate the environment.
source venvs/impute_hic2microc/bin/activate

# Normalize the .cool file if not already normalized.
cooler balance data/GM12878.GSE115524.Homo_Sapiens.CTCF.b1.chr21.mcool::resolutions/5000

# Run HiC2MicroC imputation with CPU memory/time tracking.
/usr/bin/time -v python HiC2MicroC/src/HiC2MicroC.py \
  -f1 data/GM12878.GSE115524.Homo_Sapiens.CTCF.b1.chr21.mcool::resolutions/5000 \
  -f2 data/hg38.chr21.sizes \
  -f3 impute_hic2microc_chr21_5kb_1kb_${SLURM_JOB_ID}
