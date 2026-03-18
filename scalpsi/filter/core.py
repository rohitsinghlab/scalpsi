"""
Core filtering logic for raw perturbation datasets.

Filters raw h5ad files to keep only cells whose perturbation target gene
appears in the cross-validation split files, plus non-targeting control cells
(optionally downsampled).
"""

import json
import numpy as np
import scanpy as sc
from pathlib import Path

from scalpsi import config

# Supported datasets with their perturbation column names
VALID_DATASETS = {
    "K562":    {"pert_col": "gene"},
    "RPE1":    {"pert_col": "gene"},
    "HepG2":   {"pert_col": "gene"},
    "Jurkat":  {"pert_col": "gene"},
    "HCT116":  {"pert_col": "gene_target"},
    "HEK293T": {"pert_col": "gene_target"},
}

# Variants that should be treated as non-targeting controls
CONTROL_VARIANTS = [
    "non-targeting", "ctrl", "control", "non targeting", "nontargeting",
    "negative", "neg_ctrl", "scramble", "scrambled",
]


def _load_split_genes(splits_dir: Path) -> set:
    """Load the union of all genes across all CV split files."""
    all_genes = set()
    split_files = sorted(splits_dir.glob("split*.json"))
    if not split_files:
        raise FileNotFoundError(f"No split*.json files found in {splits_dir}")
    for f in split_files:
        with open(f) as fh:
            split = json.load(fh)
        for key in ("train", "val", "test"):
            if key in split:
                all_genes.update(split[key])
    return all_genes


def _validate_dataset(dataset: str) -> dict:
    """Validate dataset name and return its metadata."""
    if dataset not in VALID_DATASETS:
        valid = ", ".join(VALID_DATASETS.keys())
        raise ValueError(
            f"Dataset '{dataset}' is not supported. "
            f"Supported datasets: {valid}."
        )
    return VALID_DATASETS[dataset]


def _is_control(series) -> "pd.Series":
    """Return boolean mask for cells that are non-targeting controls."""
    upper = series.astype(str).str.upper()
    return upper.isin([v.upper() for v in CONTROL_VARIANTS])


def filter_dataset(
    dataset: str,
    input_path: str,
    output_path: str,
    splits_dir: str = None,
    max_controls: int = 0,
    seed: int = 42,
):
    """
    Filter a raw perturbation h5ad to cells with genes in the CV splits.

    Parameters
    ----------
    dataset : str
        Dataset name. Must be one of: K562, RPE1, HepG2, Jurkat, HCT116, HEK293T.
    input_path : str
        Path to raw h5ad file.
    output_path : str
        Path to write filtered h5ad file.
    splits_dir : str, optional
        Directory containing split*.json files. Defaults to data/splits/ in repo.
    max_controls : int
        Maximum number of non-targeting control cells to keep (0 = keep all).
    seed : int
        Random seed for control subsampling.
    """
    ds_info = _validate_dataset(dataset)
    pert_col = ds_info["pert_col"]
    np.random.seed(seed)

    # Load split genes
    if splits_dir is None:
        splits_dir = config.DATA_DIR / "splits"
    else:
        splits_dir = Path(splits_dir)
    split_genes = _load_split_genes(splits_dir)
    print(f"Split genes: {len(split_genes)} unique genes across {len(list(splits_dir.glob('split*.json')))} splits")

    # Load dataset in backed mode (these are large files)
    print(f"Loading {input_path} (backed mode)...")
    adata = sc.read_h5ad(input_path, backed="r")
    print(f"  Raw shape: {adata.shape}")
    print(f"  Perturbation column: '{pert_col}'")

    # Build masks
    pert_mask = adata.obs[pert_col].isin(split_genes)
    ctrl_mask = _is_control(adata.obs[pert_col])
    print(f"  Cells with split genes: {pert_mask.sum():,}")
    print(f"  Control cells: {ctrl_mask.sum():,}")

    # Subsample controls (only if max_controls > 0)
    if max_controls > 0 and ctrl_mask.sum() > max_controls:
        ctrl_idx = np.where(ctrl_mask)[0]
        keep_ctrl_idx = set(np.random.choice(ctrl_idx, size=max_controls, replace=False))
        ctrl_mask_downsampled = np.array([
            (i in keep_ctrl_idx) for i in range(len(adata))
        ])
        print(f"  Downsampled controls: {ctrl_mask.sum():,} -> {max_controls:,}")
    else:
        ctrl_mask_downsampled = ctrl_mask.values

    keep_mask = pert_mask.values | ctrl_mask_downsampled

    # Filter and load into memory
    print(f"  Keeping {keep_mask.sum():,} cells total")
    filtered = adata[keep_mask].to_memory()

    # Standardize control labels to 'non-targeting'
    col = filtered.obs[pert_col]
    if hasattr(col, "cat"):
        if "non-targeting" not in col.cat.categories:
            filtered.obs[pert_col] = col.cat.add_categories("non-targeting")
    control_cells = _is_control(filtered.obs[pert_col])
    filtered.obs.loc[control_cells, pert_col] = "non-targeting"
    if control_cells.any():
        print(f"  Standardized {control_cells.sum():,} control labels -> 'non-targeting'")

    n_perts = filtered.obs[pert_col].nunique()
    print(f"  Final shape: {filtered.shape}, {n_perts} unique perturbations")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_h5ad(str(output))
    print(f"  Saved to {output_path}")

    return filtered
