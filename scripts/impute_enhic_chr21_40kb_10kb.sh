#!/bin/bash
#SBATCH --job-name=impute_enhic_chr21_40kb_10kb
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

module load python
conda activate env_EnHiC

cd EnHiC

CHROM=21
LEN_SIZE=400
GENOMIC_DIST=2000000
RAW_COOL_PATH="data/raw/${RAW_COOL_FILENAME}"
INPUT_MCOOL="../data/GM12878.GSE115524.Homo_Sapiens.CTCF.b1.chr21.mcool"
PRETRAINED="pretrained_model/gen_model_${LEN_SIZE}"
SAVED_MODEL="saved_model/gen_model_${LEN_SIZE}"

# Patch test_preprocessing.py to accept raw_hic from CLI,
sed -i 's|^raw_hic=.*|raw_hic = sys.argv[4] if len(sys.argv) > 4 else "Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"|' test_preprocessing.py

# Patch test_predict.py to accept raw_hic from CLI.
sed -i 's|^[[:space:]]*raw_hic = .*|    raw_hic = sys.argv[4] if len(sys.argv) > 4 else "Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"|' test_predict.py

mkdir -p data/raw
possible_resolutions=$(cooler ls $INPUT_MCOOL)
if ! echo "$possible_resolutions" | grep -q "40000"; then
  cooler coarsen -k 4 "${INPUT_MCOOL}::resolutions/10000" --append
fi

cooler balance "${INPUT_MCOOL}::resolutions/40000"

INPUT_MCOOL_40KB="${INPUT_MCOOL}::resolutions/40000"

mkdir -p "$SAVED_MODEL"
cp -r "$PRETRAINED"/* "$SAVED_MODEL/"

/usr/bin/time -v python test_predict.py "$CHROM" "$LEN_SIZE" "$GENOMIC_DIST" "$INPUT_MCOOL_40KB"
