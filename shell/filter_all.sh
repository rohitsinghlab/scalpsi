#!/bin/bash
#
# Step 0: Filter raw perturbation datasets to cells with genes in CV splits.
#
# Raw data: rawdata/perturbSeq/ (symlink to /hpc/group/singhlab/rawdata/perturbSeq/)
# Output:   data_archive/ or any directory you choose
#
# Only the three large datasets need this step:
#   K562     (~62 GB raw)  -> perturbation column: 'gene'
#   HCT116   (~195 GB raw) -> perturbation column: 'gene_target'
#   HEK293T  (~327 GB raw) -> perturbation column: 'gene_target'

RAWDATA_DIR=rawdata/perturbSeq
OUTPUT_DIR=data_archive

echo "=== Filtering raw datasets to CV split genes ==="

echo "Filtering K562..."
python scripts/filter.py \
    --dataset K562 \
    --input ${RAWDATA_DIR}/K562_raw_sc.h5ad \
    --output ${OUTPUT_DIR}/K562_filtered.h5ad

echo "Filtering HCT116..."
python scripts/filter.py \
    --dataset HCT116 \
    --input ${RAWDATA_DIR}/HCT116.h5ad \
    --output ${OUTPUT_DIR}/HCT116_filtered.h5ad

echo "Filtering HEK293T..."
python scripts/filter.py \
    --dataset HEK293T \
    --input ${RAWDATA_DIR}/HEK293T.h5ad \
    --output ${OUTPUT_DIR}/HEK293T_filtered.h5ad

echo "Done filtering all datasets!"
