#!/bin/bash
#
# Step 1: Preprocess all 6 filtered datasets.
#
# Each dataset is preprocessed independently: normalize, log-transform,
# compute HVGs at multiple thresholds, and generate DEG files.
#
# The filtering step (step 0) already ensures all datasets share the same
# 2,278 perturbation targets from the CV splits.
#
# Usage:
#   ./shell/preprocess_all.sh
#   ./shell/preprocess_all.sh --input filtered_datasets --output preprocessed
#

INPUT_DIR="filtered_datasets"
OUTPUT_DIR="preprocessed"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)  INPUT_DIR="$2";  shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--input DIR] [--output DIR]"
            echo "  --input   Directory containing filtered h5ad files (default: filtered_datasets)"
            echo "  --output  Directory for preprocessed output (default: preprocessed)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Preprocessing all datasets ==="
echo "  Input:  ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"

for dataset in K562 RPE1 HepG2 Jurkat HCT116 HEK293T; do
    echo ""
    echo "Preprocessing ${dataset}..."
    python scripts/preprocess.py \
        --path "${INPUT_DIR}/${dataset}_filtered.h5ad" \
        --name "${dataset}" \
        --output-dir "${OUTPUT_DIR}"
done

echo ""
echo "Done preprocessing all datasets!"
