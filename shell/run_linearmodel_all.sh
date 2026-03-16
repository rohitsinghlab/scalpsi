#!/bin/bash
#
# Run linearModel across all datasets and splits
# Usage: bash run_linearmodel_all.sh
#

DATASETS=("hepg2_large" "rpe1_large" "jurkat_large" "HCT_large" "K562_large" "HEK_large")
SPLITS=(0 1 2 3 4)
HVG=5000
SUFFIX=""
METHOD="trainMean"
BASE="/cwork/hl489/Pertb_benchmark/DataSet2"

# Check if any output directories already exist
EXISTING=()
for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        dir="${BASE}/${dataset}/hvg${HVG}/${METHOD}${SUFFIX}/savedModels_split${split}"
        if [ -d "$dir" ]; then
            EXISTING+=("$dir")
        fi
    done
done
if [ ${#EXISTING[@]} -gt 0 ]; then
    echo "ERROR: The following output directories already exist:"
    for d in "${EXISTING[@]}"; do
        echo "  $d"
    done
    echo "Aborting to avoid overwriting existing results."
    exit 1
fi

echo "========================================================================"
echo "Running linearModel_mean across all datasets and splits"
echo "========================================================================"
echo "Datasets: ${DATASETS[@]}"
echo "Splits:   ${SPLITS[@]}"
echo "HVG:      ${HVG}"
echo "========================================================================"
echo ""

TOTAL=$((${#DATASETS[@]} * ${#SPLITS[@]}))
CURRENT=0
FAILED=()

for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "========================================================================"
        echo "[$CURRENT/$TOTAL] Running linearModel_mean on ${dataset}, split ${split}"
        echo "========================================================================"

        if python run_all_methods.py \
            --dataset "${dataset}" \
            --methods linearModel_mean \
            --split-index "${split}" \
            --hvg "${HVG}" \
            --suffix "${SUFFIX}"; then
            echo "✓ Success: ${dataset} split ${split}"
        else
            echo "✗ Failed: ${dataset} split ${split}"
            FAILED+=("${dataset}_split${split}")
        fi
    done
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
