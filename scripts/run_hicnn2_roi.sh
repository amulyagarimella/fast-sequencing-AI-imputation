#!/bin/bash

# Input parameters

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mcool> <chrom> <chrom_len>"
    exit 1
fi

MCCOOL=$1
CHROM=$2
CHROM_LEN=$3
RESOLUTION=10000
MODEL=3
RATIO=16
OUTPUT_DIR="outputs/imputed/HiCNN2/$(date +%Y-%m-%d_%H-%M-%S)/"
mkdir -p "${OUTPUT_DIR}"

# Step 1: Extract contacts
echo "Extracting contacts..."
cooler dump --table pixels --join --header \
    ${MCCOOL}::resolutions/${RESOLUTION} \
    -r chr${CHROM}:1-${CHROM_LEN} \
    | tail -n +2 | awk '{print $2 "\t" $5 "\t" $7}' > ${OUTPUT_DIR}${CHROM}_contacts.txt

# Step 2: Generate HiCNN2 input
echo "Generating HiCNN2 input..."
python HiCNN2_package/get_HiCNN2_input_fromMat.py \
    ${OUTPUT_DIR}${CHROM}_contacts.txt \
    ${CHROM_LEN} \
    ${RESOLUTION} \
    ${OUTPUT_DIR}${CHROM}.subMats \
    ${OUTPUT_DIR}${CHROM}.index

# Step 3: Mark ROIs
# We run the predictor on input submatrices along the diagonal. If no loops predicted along the diagonal, skip that entire row + column (that's why we need to pass in indices)
# TODO
echo "Marking ROIs..."
python scripts/mark_ROIs.py \
    ${OUTPUT_DIR}${CHROM}.subMats.npy \
    ${OUTPUT_DIR}${CHROM}.index.npy \
    ${RESOLUTION} \
    ${OUTPUT_DIR}${CHROM}.subMats_ROIs

# Step 4: Run prediction
echo "Running HiCNN2 prediction..."
python HiCNN2_package/HiCNN2_predict_roi.py \
    -f1 ${OUTPUT_DIR}${CHROM}.subMats.npy \
    -f2 ${OUTPUT_DIR}${CHROM}.subMats_HiCNN2${MODEL}_${RATIO} \
    -mid ${MODEL} \
    -m HiCNN2_package/checkpoint/model_HiCNN2${MODEL}_${RATIO}.pt \
    -r ${RATIO} \
    -roi ${OUTPUT_DIR}${CHROM}.subMats_ROIs.npy

# Step 5: Combine submatrices
echo "Combining submatrices..."
gtime -f "\
user: %U s\n\
sys: %S s\n\
wall: %E\n\
cpu%%: %P\n\
majflt: %D\n\
minflt: %d\n\
maxrss: %K kb\n\
in: %I\n\
out: %O\n\
swaps: %W" \
python HiCNN2_package/combine_subMats.py \
${OUTPUT_DIR}${CHROM}.subMats_HiCNN2${MODEL}_${RATIO}.npy \
${OUTPUT_DIR}${CHROM}.index.npy \
${CHROM_LEN} \
${RESOLUTION} \
${OUTPUT_DIR}${CHROM}_predicted_hic \
> ${OUTPUT_DIR}${CHROM}_predicted_hic.out \
2> ${OUTPUT_DIR}${CHROM}_predicted_hic.err

# Step 6: Convert to sparse
echo "Converting to sparse..."
python  scripts/hic_to_sparse.py ${OUTPUT_DIR}${CHROM}_predicted_hic.npy > ${OUTPUT_DIR}${CHROM}_predicted.sparse

# Step 7: Generate chromosome sizes if needed
echo "Generating chromosome sizes..."
python scripts/get_chr_sizes.py ${MCCOOL} ${RESOLUTION} ${OUTPUT_DIR}${CHROM}.sizes

# Step 8: Convert to .cool
echo "Converting to .cool..."
cooler load -f coo \
    --count-as-float \
    --assembly hg19 \
    ${OUTPUT_DIR}${CHROM}.sizes:${RESOLUTION} \
    ${OUTPUT_DIR}${CHROM}_predicted.sparse \
    ${OUTPUT_DIR}${CHROM}_predicted.cool