#!/bin/bash
#SBATCH --job-name=compute_metrics_test
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:30
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drakedu@college.harvard.edu

# Load modules.
module load python

# Activate environment.
source venvs/fast-sequencing-AI-imputation/bin/activate

# Run the script in default test mode.
python scripts/compute_metrics.py
