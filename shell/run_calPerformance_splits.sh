#!/bin/bash
# Run calPerformance one split at a time to avoid OOM.
# Usage: bash run_calPerformance_splits.sh <dataset> [splits...]
# Example: bash run_calPerformance_splits.sh K562 5 6 7 8 9

export JAX_PLATFORMS=cpu

DATASET=${1:?Usage: bash run_calPerformance_splits.sh <dataset> [splits...]}
shift
SPLITS=("${@:-5 6 7 8 9}")

HVG=5000
WORKERS=5
METHODS="GEARS scGPT CPA trainMean controlMean"
BASE_DIR="/cwork/hl489/Pertb_benchmark/DataSet2"
SCRIPT="/cwork/hl489/perturbBench/run_calPerformance.py"

echo "Dataset:  $DATASET"
echo "Splits:   ${SPLITS[*]}"
echo "HVG:      $HVG"
echo "Workers:  $WORKERS"
echo ""

for split in "${SPLITS[@]}"; do
    echo "========== Split $split =========="
    if python "$SCRIPT" \
        --dataset "$DATASET" --hvg "$HVG" \
        --methods $METHODS \
        --splits "$split" --workers "$WORKERS" \
        --base-dir "$BASE_DIR" \
        --output "${BASE_DIR}/performance_${DATASET}_hvg${HVG}_split${split}.tsv"; then
        echo "Split $split completed successfully"
    else
        echo "WARNING: Split $split failed, continuing with remaining splits..."
    fi
    echo ""
done

# Combine all split results into one file
COMBINED="${BASE_DIR}/performance_${DATASET}_hvg${HVG}.tsv"
FIRST=true
for split in "${SPLITS[@]}"; do
    SPLIT_FILE="${BASE_DIR}/performance_${DATASET}_hvg${HVG}_split${split}.tsv"
    if [ ! -f "$SPLIT_FILE" ]; then
        echo "WARNING: $SPLIT_FILE not found, skipping"
        continue
    fi
    if $FIRST; then
        head -1 "$SPLIT_FILE" > "$COMBINED"
        FIRST=false
    fi
    tail -n +2 "$SPLIT_FILE" >> "$COMBINED"
done

echo "=========================================="
echo "Combined results saved to: $COMBINED"
echo "=========================================="
