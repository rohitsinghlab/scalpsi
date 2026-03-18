# Scalpsi

**Scalpels and Sledgehammers: Why the Mean Baseline Excels at Perturbation Prediction**

*Huan Liang and Rohit Singh — Duke University*

The mean baseline's dominance in perturbation prediction isn't a failure of model architectures — it reflects a conserved biological spectrum. "Sledgehammer" perturbations (core machinery) produce stereotyped responses the mean captures; "scalpels" (signaling regulators, epigenetic modifiers) produce gene-specific effects it misses. The Perturbation Specificity Index (PSI) formalizes this as a two-regime problem.

## Overview

This repository provides:
- **PSI**: A variance-ratio metric quantifying each perturbation's deviation from the shared response (ρ = 0.86 with baseline accuracy, Kendall's W = 0.56–0.58 across cell types)
- **Sledgehammer–Scalpel classification**: Genome-wide PSI predictions for all protein-coding genes
- **Benchmarking pipeline**: Preprocessing, method orchestration (GEARS, scGPT, CPA, GenePert, and others), and PSI-stratified evaluation
- **Reproducible analysis**: Notebooks for all paper figures

## Installation

```bash
git clone https://github.com/rohitsinghlab/scalpsi.git
cd scalpsi
pip install -e .

# For evaluation metrics (requires pertpy):
pip install -e ".[eval]"
```

## Quick Start

### 0. Filter raw data

Each raw dataset must be filtered to keep only cells whose perturbation target gene appears in the cross-validation splits (train, val, or test). The split files in `data/splits/` define 2,278 genes across 5 CV folds. Raw data lives in `rawdata/perturbSeq/`.

```bash
# Filter one dataset
python scripts/filter.py \
    --dataset K562 \
    --input rawdata/perturbSeq/K562_raw_sc.h5ad \
    --output data_archive/K562_filtered.h5ad

# Optionally downsample non-targeting controls (default: keep all)
python scripts/filter.py \
    --dataset K562 \
    --input rawdata/perturbSeq/K562_raw_sc.h5ad \
    --output data_archive/K562_filtered.h5ad \
    --max-controls 10000

# Or filter all three large datasets at once
./shell/filter_all.sh
```

### 1. Preprocess a dataset

```bash
scalpsi-preprocess --path data_archive/K562_filtered.h5ad --name K562
```

Or for multiple datasets sharing only common perturbations:

```bash
scalpsi-preprocess-shared \
    --datasets data_archive/HEK293T_filtered.h5ad:HEK293T data_archive/HCT116_filtered.h5ad:HCT116 \
    --output-dir data_archive
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
│   ├── config.py          # Path configuration
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

We evaluate four methods in the paper: **TrainMean** (linearModel_mean), **CPA**, **scGPT**, and **GenePert**. Methods are run inside podman containers from the [scPerturBench](https://github.com/bm2-lab/scPerturBench) framework ([container image on Zenodo](https://zenodo.org/records/15904698)).

> Wei, Z., Wang, Y., Gao, Y., Liu, Q. et al. Benchmarking algorithms for generalizable single-cell perturbation response prediction. *Nature Methods*, 2025.

The container provides the following conda environments, each packaging one or more methods:

| Environment | Methods |
|-------------|---------|
| cpa | CPA, scouter, biolord, inVAE, scDisInFact, scPRAM, scPreGAN, SCREEN, trVAE, cycleCDR, PRnet |
| gears | GEARS, GenePert, AttentionPert, scFoundation, scELMo, GeneCompass |
| scGPT | scGPT |
| linearModel | linearModel |
| pertpyV7 | evaluation metrics |
| cellot | CellOT |
| scarches | scGen |
| scVIDR | scVIDR |
| chemCPA | chemCPA |

## Data

Raw Perturb-seq data from three studies: Replogle et al. 2022 (K562, RPE1), Nadig et al. 2025 (HepG2, Jurkat), and X-Atlas/Orion (HCT116, HEK293T). 2,278 perturbation targets are shared across all six cell types. See Step 0 above for filtering instructions. Genome-wide PSI predictions are available in `data/`.

## Citation

If you use this code, please cite:

> Liang, H. and Singh, R. Scalpels and Sledgehammers: Why the Mean Baseline Excels at Perturbation Prediction. *Bioinformatics*, 2026. Under review at ECCB.

## License

MIT
