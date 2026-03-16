#!/bin/bash

# echo "Preprocessing hepg2..."
# python /cwork/hl489/perturbBench/example_preprocess.py \
#     --path hepg2_filtered.h5ad \
#     --name hepg2 \
#     --min-cells 0 \
#     --output-dir /cwork/hl489/Pertb_benchmark/DataSet2

# echo "Preprocessing rpe1..."
# python /cwork/hl489/perturbBench/example_preprocess.py \
#     --path rpe1_filtered.h5ad \
#     --name rpe1 \
#     --min-cells 0 \
#     --output-dir /cwork/hl489/Pertb_benchmark/DataSet2

# echo "Preprocessing jurkat..."
# python /cwork/hl489/perturbBench/example_preprocess.py \
#     --path jurkat_filtered.h5ad \
#     --name jurkat \
#     --min-cells 0 \
#     --output-dir /cwork/hl489/Pertb_benchmark/DataSet2

# echo "Done preprocessing all datasets!"


# Set to true to only keep perturbations valid across ALL three cell types
SHARED_PERTS=true

if [ "$SHARED_PERTS" = true ]; then
    echo "Preprocessing with SHARED perturbations across all cell types..."
    python /cwork/hl489/perturbBench/preprocess_shared.py \
        --datasets HEK_filtered.h5ad:HEK293T HCT_filtered.h5ad:HCT116 K562_filtered.h5ad:K562 \
        --min-cells 0 \
        --output-dir /cwork/hl489/Pertb_benchmark/DataSet2
else
    echo "Preprocessing each dataset independently..."

    echo "Preprocessing HEK293T..."
    python /cwork/hl489/perturbBench/example_preprocess.py \
        --path HEK_filtered.h5ad \
        --name HEK293T \
        --min-cells 0 \
        --output-dir /cwork/hl489/Pertb_benchmark/DataSet2

    echo "Preprocessing HCT116..."
    python /cwork/hl489/perturbBench/example_preprocess.py \
        --path HCT_filtered.h5ad \
        --name HCT116 \
        --min-cells 0 \
        --output-dir /cwork/hl489/Pertb_benchmark/DataSet2

    echo "Preprocessing K562..."
    python /cwork/hl489/perturbBench/example_preprocess.py \
        --path K562_filtered.h5ad \
        --name K562 \
        --min-cells 0 \
        --output-dir /cwork/hl489/Pertb_benchmark/DataSet2
fi

echo "Done preprocessing all datasets!"
