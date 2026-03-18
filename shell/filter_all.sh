#!/bin/bash
#
# Step 0: Filter all 6 raw perturbation datasets to cells with genes in CV splits.
#
# Usage:
#   ./shell/filter_all.sh
#   ./shell/filter_all.sh --rawdata /path/to/rawdata --output /path/to/output
#

RAWDATA_DIR="rawdata/perturbSeq"
OUTPUT_DIR="filtered_datasets"

while [[ $# -gt 0 ]]; do
    case $1 in
        --rawdata) RAWDATA_DIR="$2"; shift 2 ;;
        --output)  OUTPUT_DIR="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--rawdata DIR] [--output DIR]"
            echo "  --rawdata  Directory containing raw h5ad files (default: rawdata/perturbSeq)"
            echo "  --output   Directory for filtered output files (default: filtered_datasets)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Filtering raw datasets to CV split genes ==="
echo "  Raw data: ${RAWDATA_DIR}"
echo "  Output:   ${OUTPUT_DIR}"

for dataset_info in \
    "K562:K562_raw_sc.h5ad" \
    "RPE1:rpe1_raw_sc.h5ad" \
    "HepG2:hepg2_raw_sc.h5ad" \
    "Jurkat:jurkat_raw_sc.h5ad" \
    "HCT116:HCT116.h5ad" \
    "HEK293T:HEK293T.h5ad"; do

    dataset="${dataset_info%%:*}"
    filename="${dataset_info##*:}"

    echo "Filtering ${dataset}..."
    python scripts/filter.py \
        --dataset "${dataset}" \
        --input "${RAWDATA_DIR}/${filename}" \
        --output "${OUTPUT_DIR}/${dataset}_filtered.h5ad"
done

echo "Done filtering all datasets!"
