#!/bin/bash
#
# Run all 3 full-control analyses for HCT116.
# Outputs:
#  1) filter_hvg5000noMatch_logNor.h5ad
#  2) filter_hvg5000K562Match_logNor.h5ad
#  3) filter_hvg5000K562matchfilterfirst_logNor.h5ad
#

set -euo pipefail

DATASET="HCT116"
BASE_DIR="/cwork/hl489/Pertb_benchmark/DataSet2"
K562_REF="/cwork/hl489/Pertb_benchmark/DataSet2/K562/hvg5000/filter_hvgall_logNor.h5ad"
HVG=5000

echo "======================================================================"
echo "Running 3 analyses for ${DATASET}"
echo "======================================================================"

echo ""
echo "[1/3] noMatch: replace controls + normalize/log controls + HVG5000+pert genes (no K562 matching)"
python /cwork/hl489/perturbBench/build_fullmatched_hvg5000.py \
  --datasets "${DATASET}" \
  --dataset2-base "${BASE_DIR}" \
  --k562-ref "${K562_REF}" \
  --hvg "${HVG}" \
  --no-k562-match \
  --output-name "filter_hvg5000noMatch_logNor.h5ad"

echo ""
echo "[2/3] K562Match: same as above + K562 gene-space match before HVG"
python /cwork/hl489/perturbBench/build_fullmatched_hvg5000.py \
  --datasets "${DATASET}" \
  --dataset2-base "${BASE_DIR}" \
  --k562-ref "${K562_REF}" \
  --hvg "${HVG}" \
  --output-name "filter_hvg5000K562Match_logNor.h5ad"

echo ""
echo "[3/3] K562matchfilterfirst: counts-first + K562 gene filtering + normalize/log + HVG"
python /cwork/hl489/perturbBench/build_fullmatched_hvg5000.py \
  --datasets "${DATASET}" \
  --dataset2-base "${BASE_DIR}" \
  --k562-ref "${K562_REF}" \
  --hvg "${HVG}" \
  --counts-first \
  --output-name "filter_hvg5000K562matchfilterfirst_logNor.h5ad"

echo ""
echo "Done for ${DATASET}."
