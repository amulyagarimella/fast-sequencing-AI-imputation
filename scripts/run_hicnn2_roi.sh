#!/bin/bash

# Input parameters

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <mcool> <chrom> <chrom_len> [--roi]"
    exit 1
fi

MCCOOL=$1
CHROM=$2
CHROM_LEN=$3
USE_ROI=0
if [ "$#" -eq 4 ]; then
    if [ "$4" = "--roi" ]; then
        USE_ROI=1
    else
        echo "Unknown option: $4"
        exit 1
    fi
fi
RESOLUTION=10000
MODEL=3
RATIO=16
if [ ${USE_ROI} -eq 1 ]; then
    OUTPUT_DIR="outputs/imputed/HiCNN2/roi/$(date +%Y-%m-%d_%H-%M-%S)/"
else
    OUTPUT_DIR="outputs/imputed/HiCNN2/full/$(date +%Y-%m-%d_%H-%M-%S)/"
fi
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
if [ ${USE_ROI} -eq 1 ]; then
    # We run the predictor on input submatrices along the diagonal. If no loops predicted along the diagonal, skip that entire row + column (that's why we need to pass in indices)
    echo "Marking ROIs..."
    python scripts/mark_ROIs.py \
        ${OUTPUT_DIR}${CHROM}.subMats.npy \
        ${OUTPUT_DIR}${CHROM}.index.npy \
        ${RESOLUTION} \
        ${OUTPUT_DIR}${CHROM}.subMats_ROIs
fi

# Step 4: Run prediction
ROIS_TO_INPUT=""
if [ ${USE_ROI} -eq 1 ]; then
    ROIS_TO_INPUT="--roi-indices ${OUTPUT_DIR}${CHROM}.subMats_ROIs.npy"
fi
echo "Running HiCNN2 prediction..."
gtime -f "\
%C   command line and arguments\n \
%c   involuntary context switches\n \
%E   elapsed real time (wall clock) in [hour:]min:sec\n \
%e   elapsed real time (wall clock) in seconds\n \
%F   major page faults\n \
%M   maximum resident set size in KB\n \
%P   percent of CPU this job got\n \
%R   minor page faults\n \
%S   system (kernel) time in seconds\n \
%U   user time in seconds\n \
%w   voluntary context switches\n \
%x   exit status of command" \
python HiCNN2_package/HiCNN2_predict_roi.py \
    -f1 ${OUTPUT_DIR}${CHROM}.subMats.npy \
    -f2 ${OUTPUT_DIR}${CHROM}.subMats_HiCNN2${MODEL}_${RATIO} \
    -mid ${MODEL} \
    -m HiCNN2_package/checkpoint/model_HiCNN2${MODEL}_${RATIO}.pt \
    -r ${RATIO} \
    ${ROIS_TO_INPUT} \
> ${OUTPUT_DIR}${CHROM}_predicted_hic.out \
2> ${OUTPUT_DIR}${CHROM}_predicted_hic.err

# Step 5: Combine submatrices
echo "Combining submatrices..."
python HiCNN2_package/combine_subMats.py \
${OUTPUT_DIR}${CHROM}.subMats_HiCNN2${MODEL}_${RATIO}.npy \
${OUTPUT_DIR}${CHROM}.index.npy \
${CHROM_LEN} \
${RESOLUTION} \
${OUTPUT_DIR}${CHROM}_predicted_hic \

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