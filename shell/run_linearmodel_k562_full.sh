#!/bin/bash
#
# Run trainMean_full baseline for K562 only, using filter_hvg5000full_logNor.h5ad
# Usage: bash run_linearmodel_k562_full.sh
#

set -e

DATASET="K562"
SPLITS=(5 6 7 8 9)
HVG=5000
INPUT_FILE="filter_hvg5000full_logNor.h5ad"
SCENARIO="trainMean_full"
SCRIPT="/cwork/hl489/perturbBench/linearModel_trainMean_full.py"

echo "========================================================================"
echo "Running trainMean_full for K562 only"
echo "========================================================================"
echo "Dataset:    ${DATASET}"
echo "Splits:     ${SPLITS[@]}"
echo "HVG:        ${HVG}"
echo "Input file: ${INPUT_FILE}"
echo "Scenario:   ${SCENARIO}"
echo "Script:     ${SCRIPT}"
echo "========================================================================"
echo ""

TOTAL=${#SPLITS[@]}
CURRENT=0
FAILED=()

for split in "${SPLITS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================================================"
    echo "[$CURRENT/$TOTAL] Running ${DATASET}, split ${split}"
    echo "========================================================================"

    if python "${SCRIPT}" \
        --dataset "${DATASET}" \
        --split-index "${split}" \
        --hvg "${HVG}" \
        --input-file "${INPUT_FILE}" \
        --scenario "${SCENARIO}"; then
        echo "✓ Success: ${DATASET} split ${split}"
    else
        echo "✗ Failed: ${DATASET} split ${split}"
        FAILED+=("${DATASET}_split${split}")
    fi
done

echo ""
echo "========================================================================"
echo "FINAL SUMMARY"
echo "========================================================================"
echo "Total runs:     ${TOTAL}"
echo "Successful:     $((TOTAL - ${#FAILED[@]}))"
echo "Failed:         ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    for item in "${FAILED[@]}"; do
        echo "  - ${item}"
    done
    exit 1
else
    echo ""
    echo "✓ All runs completed successfully!"
    exit 0
fi
