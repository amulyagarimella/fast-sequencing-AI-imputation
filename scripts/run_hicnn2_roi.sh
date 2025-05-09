#!/bin/bash

# Initialize default values
USE_ROI=0
ROI_SPARSITY=0.1
ROI_METHOD="ridge"
INTERPOLATION="lowres"
EXPECTED_VALUES=""
RESOLUTION=10000
MODEL=3
RATIO=16
APPLY_DOWN_RATIO=0

# Parse all arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r) USE_ROI=1; shift ;;
        -s) ROI_SPARSITY="$2"; shift 2 ;;
        -m) ROI_METHOD="$2"; shift 2 ;;
        -i) INTERPOLATION="$2"; shift 2 ;;
        -e) EXPECTED_VALUES="$2"; shift 2 ;;
        -a) APPLY_DOWN_RATIO=1; shift ;;
        -M) MODEL="$2"; shift 2 ;;
        -R) RESOLUTION="$2"; shift 2 ;;
        --ratio) RATIO="$2"; shift 2 ;;
        --) shift; break ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

# Set positional arguments
set -- "${POSITIONAL[@]}"

# Validate arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <mcool> <chrom_num> <chrom_len> [options...]"
    echo "Options can be in any order:"
    echo "  -r              Enable ROI mode"
    echo "  -s <sparsity>   ROI sparsity threshold (0.0-1.0)"
    echo "  -m <method>     ROI detection method"
    echo "  -i <method>     Interpolation method"
    echo "  -e <file>       Expected values TSV"
    echo "  -a              Apply downscaling ratio"
    echo "  -M <model>      HiCNN2 model version"
    echo "  -R <resolution> Resolution in bp"
    echo "  --ratio <value> Downscaling ratio"
    exit 1
fi

MCCOOL=$1
CHROM=$2
CHROM_LEN=$3

# Validate interpolation method
if [ "$INTERPOLATION" = "expected" ] && [ -z "$EXPECTED_VALUES" ]; then
    echo "Error: --expected must be specified when using 'expected' interpolation"
    exit 1
fi

if [ ${USE_ROI} -eq 1 ]; then
    if [ ${RATIO} -eq 1 ]; then
        OUTPUT_DIR="outputs/imputed/HiCNN2/original/roi/${INTERPOLATION}/$(date +%Y-%m-%d_%H-%M-%S)/"
    else
        OUTPUT_DIR="outputs/imputed/HiCNN2/down${RATIO}/roi/${INTERPOLATION}/$(date +%Y-%m-%d_%H-%M-%S)/"
    fi
else
    if [ ${RATIO} -eq 1 ]; then
        OUTPUT_DIR="outputs/imputed/HiCNN2/original/full/$(date +%Y-%m-%d_%H-%M-%S)/"
    else
        OUTPUT_DIR="outputs/imputed/HiCNN2/down${RATIO}/full/$(date +%Y-%m-%d_%H-%M-%S)/"
    fi
fi

mkdir -p "${OUTPUT_DIR}"

# Log parameters
echo "Parameters:" > "${OUTPUT_DIR}parameters.txt"
echo "MCCOOL: ${MCCOOL}" >> "${OUTPUT_DIR}parameters.txt"
echo "CHROM: ${CHROM}" >> "${OUTPUT_DIR}parameters.txt"
echo "CHROM_LEN: ${CHROM_LEN}" >> "${OUTPUT_DIR}parameters.txt"
echo "USE_ROI: ${USE_ROI}" >> "${OUTPUT_DIR}parameters.txt"
echo "ROI_SPARSITY: ${ROI_SPARSITY}" >> "${OUTPUT_DIR}parameters.txt"
echo "ROI_METHOD: ${ROI_METHOD}" >> "${OUTPUT_DIR}parameters.txt"
echo "INTERPOLATION: ${INTERPOLATION}" >> "${OUTPUT_DIR}parameters.txt"
echo "EXPECTED_VALUES: ${EXPECTED_VALUES}" >> "${OUTPUT_DIR}parameters.txt"
echo "RESOLUTION: ${RESOLUTION}" >> "${OUTPUT_DIR}parameters.txt"
echo "MODEL: ${MODEL}" >> "${OUTPUT_DIR}parameters.txt"
echo "RATIO: ${RATIO}" >> "${OUTPUT_DIR}parameters.txt" 
echo "APPLY_DOWN_RATIO: ${APPLY_DOWN_RATIO}" >> "${OUTPUT_DIR}parameters.txt"
echo "DATE: $(date)" >> "${OUTPUT_DIR}parameters.txt"

# Step 1: Extract contacts
echo "Extracting contacts..."
cooler dump --table pixels --join --header \
    ${MCCOOL} \
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
        ${OUTPUT_DIR}${CHROM}.subMats_ROIs \
        --sparsity ${ROI_SPARSITY} \
        --method ${ROI_METHOD} \
        ${APPLY_DOWN_RATIO:+--downsample ${RATIO}}
fi

# Step 4: Run prediction
ROIS_TO_INPUT=""
if [ ${USE_ROI} -eq 1 ]; then
    ROIS_TO_INPUT="--roi-indices ${OUTPUT_DIR}${CHROM}.subMats_ROIs.npy --non-roi-method ${INTERPOLATION}"
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
    -m HiCNN2_package/checkpoint/model_HiCNN2${MODEL}_${RATIO}.pt \
    -r ${RATIO} \
    --model ${MODEL} \
    --submat-indices ${OUTPUT_DIR}${CHROM}.index.npy \
    --resolution ${RESOLUTION} \
    --non-roi-method ${INTERPOLATION} \
    ${EXPECTED_VALUES:+--expected-values "$EXPECTED_VALUES"} \
    ${APPLY_DOWN_RATIO:+--apply-down-ratio} \
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

# Step 6: Convert to .cool
echo "Converting to .cool..."
python scripts/npy_to_cool.py \
    ${OUTPUT_DIR}${CHROM}_predicted_hic.npy \
    ${OUTPUT_DIR}${CHROM}_predicted.cool \
    --chrom "chr${CHROM}" \
    --resolution ${RESOLUTION}