#!/bin/bash

MCCOOL="/Users/amulyagarimella/Documents/2241finalproject/data/GM12878.GSE115524/processed/GM12878.GSE115524.Homo_Sapiens.CTCF.b1.ds.16.chr21.cool::/"
CHROM=21
CHROM_LEN=46709983
EXPECTED_FILE="/Users/amulyagarimella/Documents/2241finalproject/data/GM12878.GSE115524/processed/GM12878.GSE115524.Homo_Sapiens.CTCF.b1.expected.tsv"

for MODEL in {1..3}; do 
    echo "Running HiCNN2 with HiCNN2 model $MODEL"
        
    if [ $MODEL -ne 1 ]; then
        # Full version (non-ROI)
        bash scripts/run_hicnn2_roi.sh "$MCCOOL" "$CHROM" "$CHROM_LEN" -a -M "$MODEL" -R 10000
    else 
        echo "Skipping HiCNN2 model 1"
    fi

    for ROIMODEL in ridge random elasticnet lasso logreg; do 
        for SPARSITY in 0.1 0.25 0.5; do 
            # Lowres interpolation
            echo "Running with ROI method $ROIMODEL, sparsity $SPARSITY, lowres interpolation, model $MODEL"
            bash scripts/run_hicnn2_roi.sh "$MCCOOL" "$CHROM" "$CHROM_LEN" -a -M "$MODEL" -R 10000 -r -s "$SPARSITY" -m "$ROIMODEL" -i lowres
            
            # Expected interpolation
            echo "Running with ROI method $ROIMODEL, sparsity $SPARSITY, expected interpolation, model $MODEL"
            bash scripts/run_hicnn2_roi.sh "$MCCOOL" "$CHROM" "$CHROM_LEN" -a -M "$MODEL" -R 10000 -r -s "$SPARSITY" -m "$ROIMODEL" -i expected -e "$EXPECTED_FILE"
        done
    done
done
