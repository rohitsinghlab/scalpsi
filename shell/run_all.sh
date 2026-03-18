#!/bin/bash
#
# Step 2: Run all methods across all datasets and splits.
#
# Must be run inside the Apptainer container:
#   apptainer shell --nv --bind $PWD:/home/project/scalpsi scperturbench_v1.sif
#   source /usr/local/anaconda3/etc/profile.d/conda.sh
#   cd /home/project/scalpsi
#   bash shell/run_all.sh
#
# Usage:
#   ./shell/run_all.sh                          # all datasets, all methods, all splits
#   ./shell/run_all.sh --methods GEARS CPA      # specific methods only
#   ./shell/run_all.sh --datasets K562 RPE1     # specific datasets only
#

DATASETS=(K562 RPE1 HepG2 Jurkat HCT116 HEK293T)
METHODS=()
SPLITS=(0 1 2 3 4)

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) shift; DATASETS=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do DATASETS+=("$1"); shift; done ;;
        --methods)  shift; METHODS=();  while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do METHODS+=("$1");  shift; done ;;
        --splits)   shift; SPLITS=();   while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do SPLITS+=("$1");   shift; done ;;
        -h|--help)
            echo "Usage: $0 [--datasets D1 D2 ...] [--methods M1 M2 ...] [--splits 0 1 ...]"
            echo "  --datasets  Datasets to process (default: all 6)"
            echo "  --methods   Methods to run (default: all)"
            echo "  --splits    Split indices (default: 0 1 2 3 4)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build methods arg
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS_ARG=""
else
    METHODS_ARG="--methods ${METHODS[*]}"
fi

echo "=== Running methods ==="
echo "  Datasets: ${DATASETS[*]}"
echo "  Methods:  ${METHODS[*]:-all}"
echo "  Splits:   ${SPLITS[*]}"

TOTAL=$((${#DATASETS[@]} * ${#SPLITS[@]}))
CURRENT=0
FAILED=()

for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "========================================================================"
        echo "[$CURRENT/$TOTAL] ${dataset}, split ${split}"
        echo "========================================================================"

        if python scripts/run_methods.py \
            --dataset "${dataset}" \
            --split-index "${split}" \
            --hvg 5000 \
            ${METHODS_ARG}; then
            echo "Success: ${dataset} split ${split}"
        else
            echo "Failed: ${dataset} split ${split}"
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
    echo "All runs completed successfully!"
    exit 0
fi
