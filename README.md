# scalpsi

**Scalpels and Sledgehammers: Why Mean Baselines Dominate Perturbation Prediction**

*Rohit Singh Lab*

A benchmarking framework for evaluating genetic perturbation prediction methods in single-cell transcriptomics. Compares 11+ methods across 6 cell lines using standardized preprocessing, 5-fold cross-validation, and 9 distance metrics.

## Overview

This repository provides:
- **Preprocessing**: Normalize, log-transform, select HVGs, compute DEGs from raw h5ad datasets
- **Method orchestration**: Run GEARS, scGPT, scFoundation, CPA, and 7+ other methods with unified I/O
- **Evaluation**: Per-perturbation and per-gene metrics using `pertpy` distance functions
- **Analysis**: Cross-cell-line consistency, SVD residual analysis, gene-perturbation interaction modeling
- **Figures**: Notebooks for all paper figures

## Installation

```bash
git clone https://github.com/rohitsinghlab/scalpsi.git
cd scalpsi
pip install -e .

# For evaluation metrics (requires pertpy):
pip install -e ".[eval]"
```

## Configuration

Set environment variables to point to your data and method scripts:

```bash
export SCALPSI_DATA_DIR=/path/to/DataSet2          # processed datasets output
export SCALPSI_BASE_DIR=/path/to/Pertb_benchmark   # benchmark root
export SCALPSI_SCRIPT_DIR=/path/to/method/scripts  # GEARS, scGPT, etc.
```

## Quick Start

### 0. Filter raw data

The three large raw datasets (K562, HCT116, HEK293T) must be filtered to keep only cells with perturbation genes that appear in the cross-validation splits, plus up to 10,000 non-targeting control cells. Raw data lives in `rawdata/perturbSeq/`.

```bash
# Filter one dataset
python scripts/filter.py \
    --dataset K562 \
    --input rawdata/perturbSeq/K562_raw_sc.h5ad \
    --output data_archive/K562_filtered.h5ad

# Or filter all three at once
./shell/filter_all.sh
```

The split files in `data/splits/` define which genes to keep (2,278 genes across 5 CV folds). Only K562, HCT116, and HEK293T are supported — the smaller datasets (rpe1, jurkat, hepg2) are already pre-filtered.

### 1. Preprocess a dataset

```bash
scalpsi-preprocess --path data_archive/K562_filtered.h5ad --name K562
```

Or for multiple datasets sharing only common perturbations:

```bash
scalpsi-preprocess-shared \
    --datasets data_archive/HEK293T_filtered.h5ad:HEK293T data_archive/HCT116_filtered.h5ad:HCT116 \
    --output-dir $SCALPSI_DATA_DIR
```

### 2. Run prediction methods

```bash
# Run GEARS on split 0
scalpsi-run --dataset MyDataset --methods GEARS --hvg 5000 --split-index 0

# Run all methods
scalpsi-run --dataset MyDataset --hvg 5000 --split-index 0
```

### 3. Evaluate predictions

```bash
# Perturbation-level metrics
scalpsi-evaluate --dataset MyDataset --hvg 5000 --methods GEARS scGPT linearModel_mean

# Gene-level metrics
scalpsi-evaluate-genes --dataset MyDataset --hvg 5000 --methods GEARS scGPT
```

### 4. Analyze results

Open `notebooks/analysis.ipynb` for cross-dataset performance analysis and figure generation.

## Repository Structure

```
scalpsi/
├── scalpsi/               # Python package
│   ├── config.py          # Centralized path configuration (env vars)
│   ├── filter/            # Filter raw datasets to CV split genes
│   ├── preprocess/        # Data preprocessing (normalize, HVG, DEG)
│   ├── methods/           # Method orchestration (GEARS, scGPT, etc.)
│   ├── evaluation/        # Performance metrics (perturbation & gene level)
│   └── analysis/          # Statistical analysis utilities
│
├── scripts/               # CLI entry points
├── notebooks/             # Analysis notebooks
│   └── figures/           # Paper figure notebooks
├── data/                  # Reference files (gene_info, splits, TF lists)
│   └── splits/            # 5 train/test splits (split0.json–split4.json)
└── shell/                 # HPC job scripts
```

## Methods

| Method | Conda env | Needs GEARS? |
|--------|-----------|-------------|
| GEARS | gears | No |
| scFoundation | gears | Yes |
| scGPT | scGPT | No |
| GenePert | gears | No |
| GeneCompass | gears | Yes |
| AttentionPert | gears | Yes |
| scELMo | gears | Yes |
| scouter | cpa | Yes |
| baseMLP | cpa | Yes |
| baseReg | cpa | Yes |
| linearModel | linearModel | Yes |
| linearModel_mean | gears | No |
| CPA | cpa | No |

## Data

Processed datasets are not included (1–33 GB each). See `data/README.md` for download instructions and expected directory structure.

## Citation

If you use this code, please cite:

> [Paper title and citation to be added upon publication]

## License

MIT
