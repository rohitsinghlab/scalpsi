#!/bin/bash
#
# Step 1: Preprocess all 6 filtered datasets.
#
# Small datasets (RPE1, Jurkat, HepG2) are preprocessed independently.
# Large datasets (K562, HCT116, HEK293T) are preprocessed with shared
# perturbations only (intersection across all three).
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

# Small datasets: preprocess independently
for dataset in RPE1 Jurkat HepG2; do
    echo ""
    echo "Preprocessing ${dataset}..."
    python scripts/preprocess.py \
        --path "${INPUT_DIR}/${dataset}_filtered.h5ad" \
        --name "${dataset}" \
        --output-dir "${OUTPUT_DIR}"
done

# Large datasets: preprocess with shared perturbations
echo ""
echo "Preprocessing K562, HCT116, HEK293T with shared perturbations..."
python scripts/preprocess_shared.py \
    --datasets \
        "${INPUT_DIR}/K562_filtered.h5ad:K562" \
        "${INPUT_DIR}/HCT116_filtered.h5ad:HCT116" \
        "${INPUT_DIR}/HEK293T_filtered.h5ad:HEK293T" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Done preprocessing all datasets!"
