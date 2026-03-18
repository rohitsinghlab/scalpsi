"""
Preprocessing script for perturbation datasets.
Computes normalization, log1p, and HVGs at multiple thresholds.
Saves results in the preprocessed/ directory.

Based on myUtil1.py preData() function with extensions for multiple HVG thresholds.

Output directory defaults to preprocessed/ in the repo root.
Pass output_base_dir to override.
"""

import subprocess, os, sys
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
from scipy import sparse
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

from scalpsi import config


def get_valid_perturbations(path):
    """
    Load a dataset in backed='r' mode and return the set of perturbations.
    Warns about perturbations whose target genes are missing from .var but keeps them.
    """
    print(f"  Loading {path} in backed mode...")
    adata = sc.read_h5ad(path, backed='r')

    # Get perturbation column
    if 'perturbation' in adata.obs.columns:
        perts = adata.obs['perturbation'].copy()
    elif 'gene' in adata.obs.columns:
        perts = adata.obs['gene'].copy()
    elif 'gene_target' in adata.obs.columns:
        perts = adata.obs['gene_target'].copy()
    else:
        raise ValueError(f"No perturbation column found in {path}")

    if perts.dtype.name == 'category':
        perts = perts.astype(str)

    # Standardize control naming
    control_variants = ['ctrl', 'control', 'non-targeting', 'non targeting', 'nontargeting',
                       'negative', 'neg_ctrl', 'scramble', 'scrambled']
    for variant in control_variants:
        mask = perts.str.lower() == variant
        if mask.any():
            perts[mask] = 'control'

    # Get unique non-control perturbations
    unique_perts = set(perts.unique())
    unique_perts.discard('control')
    unique_perts.discard('None')

    # Check which perturbations have target genes missing from .var (warn only)
    var_names = set(adata.var_names)
    has_gene_name = 'gene_name' in adata.var.columns
    if has_gene_name:
        gene_names = set(adata.var['gene_name'].values)

    missing_count = 0
    for p in unique_perts:
        genes_in_pert = [g.strip() for g in p.split('+') if g.strip() not in ['ctrl', 'control']]
        for gene_symbol in genes_in_pert:
            if has_gene_name:
                gene_found = gene_symbol in gene_names
            else:
                gene_found = gene_symbol in var_names
            if not gene_found:
                missing_count += 1
                print(f"    CAREFUL: perturbation '{p}' target gene '{gene_symbol}' not in .var (kept)")
                break

    print(f"  {len(adata.obs)} cells, {len(adata.var)} genes")
    print(f"  {len(unique_perts)} unique perturbations, {missing_count} with target genes missing from .var")

    adata.file.close()
    return unique_perts, unique_perts


def compute_shared_perturbations(datasets):
    """
    Find perturbations that are valid (target genes in .var) across ALL datasets.

    Parameters
    ----------
    datasets : list of (path, name) tuples

    Returns
    -------
    shared_perts : set of perturbation names valid in all datasets
    """
    print("="*60)
    print("Computing shared perturbations across datasets")
    print("="*60)

    all_valid = []
    names = []
    for path, name in datasets:
        valid, total = get_valid_perturbations(path)
        print(f"  {name}: {len(valid)}/{len(total)} perturbations have target genes in .var")
        all_valid.append(valid)
        names.append(name)

    shared_perts = set.intersection(*all_valid)

    # Show what each dataset lost in the intersection
    print(f"\n--- Intersection result ---")
    for i, name in enumerate(names):
        dropped = all_valid[i] - shared_perts
        if dropped:
            print(f"  {name}: dropping {len(dropped)} perturbations not shared: {sorted(dropped)}")
        else:
            print(f"  {name}: all {len(all_valid[i])} valid perturbations are in the shared set")

    print(f"\nShared valid perturbations across all {len(datasets)} datasets: {len(shared_perts)}")
    print("="*60)
    return shared_perts


def preData_multiHVG(adata, domaxNumsPerturb=0, domaxNumsControl=0, minNums=0, min_cells=10, allowed_perturbations=None):
    """
    Preprocessing function that performs QC, normalization, log1p, and HVG selection.

    Based on the exact preprocessing from myUtil1.py with added HVG thresholds.

    Parameters:
    -----------
    adata : AnnData
        Input data with adata.obs['perturbation'] column required
    domaxNumsPerturb : int
        Maximum cells per perturbation (0 = no downsampling)
    domaxNumsControl : int
        Maximum control cells (0 = no downsampling)
    minNums : int
        Minimum cells per perturbation to keep
    min_cells : int
        Minimum cells expressing a gene to keep the gene
    allowed_perturbations : set or None
        If provided, only keep cells with these perturbations (+ control).
        Used for cross-dataset shared perturbation mode.

    Returns:
    --------
    adata : AnnData
        Preprocessed data with layers and HVG annotations
    """
    print("Starting preprocessing...")

    # 1. Basic cleanup
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata = adata[~adata.obs.index.duplicated()].copy()
    adata = adata[adata.obs["perturbation"] != "None"]
    filterNoneNums = adata.shape[0]
    print(f"After filtering 'None': {filterNoneNums} cells")

    # Normalize control naming - GEARS expects 'control'
    # Convert to string type if categorical to allow assignment
    if adata.obs['perturbation'].dtype.name == 'category':
        adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)

    control_variants = ['ctrl', 'control', 'non-targeting', 'non targeting', 'nontargeting',
                       'negative', 'neg_ctrl', 'scramble', 'scrambled']
    for variant in control_variants:
        mask = adata.obs['perturbation'].str.lower() == variant
        if mask.any():
            n_renamed = mask.sum()
            adata.obs.loc[mask, 'perturbation'] = 'control'
            print(f"Renamed {n_renamed} '{variant}' cells to 'control'")

    # 1b. Filter to allowed perturbations (for shared cross-dataset mode)
    if allowed_perturbations is not None:
        allowed = set(allowed_perturbations) | {'control'}
        adata = adata[adata.obs['perturbation'].isin(allowed), :]
        print(f"After shared perturbation filtering: {adata.shape[0]} cells")
        print(f"  Kept {len(allowed_perturbations)} perturbations + control")

    # 2. Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    filterCells = adata.shape[0]
    print(f"After cell/gene filtering: {filterCells} cells, {adata.shape[1]} genes")

    # 3. Mitochondrial filtering
    if np.any([i.startswith('mt-') for i in adata.var_names]):
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
    else:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    if sum(adata.obs['pct_counts_mt'] < 10) / adata.shape[0] <= 0.5:
        adata = adata[adata.obs.pct_counts_mt < 15, :]
        print("Using MT threshold: 15%")
    else:
        adata = adata[adata.obs.pct_counts_mt < 10, :]
        print("Using MT threshold: 10%")

    filterMT = adata.shape[0]
    print(f"After MT filtering: {filterMT} cells")

    # 4. Filter by perturbation frequency
    tmp = adata.obs['perturbation'].value_counts()
    tmp_bool = tmp >= minNums
    genes = list(tmp[tmp_bool].index)
    if 'control' not in genes:
        genes += ['control']
    adata = adata[adata.obs['perturbation'].isin(genes), :]
    filterMinNums = adata.shape[0]
    print(f"After min perturbation filtering (>={minNums} cells): {filterMinNums} cells")
    print(f"Number of perturbations kept: {len(genes)}")

    # 4b. Check perturbations where target genes are missing from dataset
    print("\nChecking perturbation target genes...")
    missing_count = 0
    for p in adata.obs['perturbation'].unique():
        if p == 'control':
            continue
        genes_in_pert = [g.strip() for g in p.split('+') if g.strip() not in ['ctrl', 'control']]
        for gene_symbol in genes_in_pert:
            gene_found = False
            if 'gene_name' in adata.var.columns:
                gene_found = gene_symbol in adata.var['gene_name'].values
            else:
                gene_found = gene_symbol in adata.var_names

            if not gene_found:
                missing_count += 1
                print(f"  CAREFUL: perturbation '{p}' target gene '{gene_symbol}' is not in .var")
                break

    if missing_count == 0:
        print("  All perturbation target genes found in dataset")
    else:
        print(f"  {missing_count} perturbations have target genes missing from .var (kept in dataset)")

    # 5. Optional downsampling
    if domaxNumsPerturb:
        print(f"Downsampling perturbations to max {domaxNumsPerturb} cells each...")
        adata1 = adata[adata.obs['perturbation'] == 'control']
        perturbations = adata.obs['perturbation'].unique()
        perturbations = [i for i in perturbations if i != 'control']
        adata_list = []
        for perturbation in perturbations:
            adata_tmp = adata[adata.obs['perturbation'] == perturbation]
            if adata_tmp.shape[0] > domaxNumsPerturb:
                sampled_indices = np.random.choice(adata_tmp.n_obs, domaxNumsPerturb, replace=False)
                adata_tmp = adata_tmp[sampled_indices, :]
            adata_list.append(adata_tmp)
        adata2 = ad.concat(adata_list)
        adata = ad.concat([adata1, adata2])
        adata.var = adata1.var.copy()

    if domaxNumsControl:
        print(f"Downsampling control to max {domaxNumsControl} cells...")
        adata1 = adata[adata.obs['perturbation'] == 'control']
        adata2 = adata[adata.obs['perturbation'] != 'control']
        if adata1.shape[0] > domaxNumsControl:
            sampled_indices = np.random.choice(adata1.n_obs, domaxNumsControl, replace=False)
            adata1 = adata1[sampled_indices, :]
        adata = ad.concat([adata1, adata2])
        adata.var = adata1.var.copy()

    # 6. Save raw counts
    adata.layers['counts'] = adata.X.copy()
    print("Saved raw counts to adata.layers['counts']")

    # 7. Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("Normalized to 10,000 counts per cell")

    sc.pp.log1p(adata)
    print("Applied log1p transformation")

    adata.layers['logNor'] = adata.X.copy()
    print("Saved log-normalized data to adata.layers['logNor']")

    # 8. Compute HVGs at multiple thresholds
    hvg_thresholds = [100, 1000, 2000, 5000]
    print("\nComputing highly variable genes...")
    for n_genes in hvg_thresholds:
        if n_genes <= adata.shape[1]:  # Only if we have enough genes
            sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, subset=False)
            adata.var[f'highly_variable_{n_genes}'] = adata.var['highly_variable']
            print(f"  - Top {n_genes} HVGs marked in adata.var['highly_variable_{n_genes}']")
        else:
            print(f"  - Skipping {n_genes} HVGs (only {adata.shape[1]} genes available)")

    # 9. Sort by perturbation
    adata = adata[adata.obs.sort_values(by='perturbation').index, :]

    print(f"\nFinal dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    return adata


# Alias kept for backwards compatibility
preprocess_dataset = preData_multiHVG


def calDEG(adata, condition_column='perturbation', control_tag='control'):
    """
    Compute differentially expressed genes for each perturbation vs control.
    Based on myUtil1.py calDEG() function.

    Returns a dict: {perturbation_name: DataFrame with ranked genes}
    """
    adata = adata.copy()
    adata.uns['log1p'] = {}
    adata.uns['log1p']['base'] = None
    mydict = defaultdict(dict)
    perturbations = adata.obs[condition_column].unique()
    perturbations = [i for i in perturbations if i != control_tag]
    sc.tl.rank_genes_groups(adata, condition_column, groups=perturbations, reference=control_tag, method='t-test')
    result = adata.uns['rank_genes_groups']
    for perturbation in perturbations:
        final_result = pd.DataFrame({key: result[key][perturbation] for key in ['names', 'pvals_adj', 'logfoldchanges', 'scores']})
        final_result['foldchanges'] = 2 ** final_result['logfoldchanges']
        final_result.drop(labels=['logfoldchanges'], inplace=True, axis=1)
        final_result.set_index('names', inplace=True)
        final_result['abs_scores'] = np.abs(final_result['scores'])
        final_result.sort_values('abs_scores', ascending=False, inplace=True)
        mydict[perturbation] = final_result
    return mydict


def save_hvg_subsets(adata, dataset_name, output_base_dir=None, save_datasets_for=None):
    """
    Compute HVG subsets and save DEG files (and optionally h5ad datasets).

    Parameters:
    -----------
    adata : AnnData
        Preprocessed data with HVG annotations
    dataset_name : str
        Name of the dataset (e.g., 'MyDataset')
    output_base_dir : str or None
        Base directory for output. Defaults to config.DATASET2_DIR.
    save_datasets_for : list of int or None
        Which HVG thresholds to save h5ad files for. None = [5000], [] = none.
    """
    if output_base_dir is None:
        output_base_dir = str(config.DATASET2_DIR)
    if save_datasets_for is None:
        save_datasets_for = [5000]
    hvg_thresholds = [100, 1000, 2000, 5000]

    # Ensure dataset output dir exists
    os.makedirs(os.path.join(output_base_dir, dataset_name), exist_ok=True)

    for n_genes in hvg_thresholds:
        hvg_col = f'highly_variable_{n_genes}'

        if hvg_col not in adata.var.columns:
            print(f"Skipping {n_genes} HVGs (not computed)")
            continue

        # Build HVG mask, always including perturbation target genes
        hvg_mask = adata.var[hvg_col].copy()
        pert_genes_added = []
        for p in adata.obs['perturbation'].unique():
            if p == 'control':
                continue
            for gene_symbol in p.split('+'):
                gene_symbol = gene_symbol.strip()
                if gene_symbol == 'ctrl':
                    continue

                gene_id = None
                if 'gene_name' in adata.var.columns:
                    matching = adata.var[adata.var['gene_name'] == gene_symbol].index
                    if len(matching) > 0:
                        gene_id = matching[0]
                else:
                    if gene_symbol in adata.var_names:
                        gene_id = gene_symbol

                if gene_id is not None and gene_id in adata.var_names and not hvg_mask[gene_id]:
                    hvg_mask[gene_id] = True
                    pert_genes_added.append(f"{gene_symbol} ({gene_id})")
        if pert_genes_added:
            print(f"  Added {len(pert_genes_added)} perturbation genes to hvg{n_genes} set")

        adata_hvg = adata[:, hvg_mask].copy()

        if n_genes in save_datasets_for:
            output_dir = os.path.join(output_base_dir, dataset_name, f"hvg{n_genes}")
            os.makedirs(output_dir, exist_ok=True)

            output_file_all = os.path.join(output_dir, "filter_hvgall_logNor.h5ad")
            adata.write_h5ad(output_file_all)
            print(f"Saved full dataset: {output_file_all}")

            output_file_hvg = os.path.join(output_dir, f"filter_hvg{n_genes}_logNor.h5ad")
            adata_hvg.write_h5ad(output_file_hvg)
            print(f"Saved HVG subset ({adata_hvg.shape[1]} genes): {output_file_hvg}")

        # Generate DEG file for this HVG subset
        print(f"  Computing DEGs for hvg{n_genes}...")
        deg_dict = calDEG(adata_hvg)
        deg_file = os.path.join(output_base_dir, dataset_name, f"DEG_hvg{n_genes}.pkl")
        with open(deg_file, 'wb') as fout:
            pickle.dump(deg_dict, fout)
        print(f"  Saved DEG file: {deg_file}")


def preprocess_and_save(input_file, dataset_name,
                       minNums=50,
                       domaxNumsPerturb=0,
                       domaxNumsControl=0,
                       output_base_dir=None,
                       allowed_perturbations=None,
                       save_datasets_for=None):
    """
    Complete preprocessing pipeline: load -> preprocess -> save.

    Parameters:
    -----------
    input_file : str
        Path to input h5ad file
    dataset_name : str
        Name for the dataset (used in output paths)
    minNums : int
        Minimum cells per perturbation
    domaxNumsPerturb : int
        Max cells per perturbation (0 = no limit)
    domaxNumsControl : int
        Max control cells (0 = no limit)
    output_base_dir : str or None
        Base directory for output. Defaults to config.DATASET2_DIR
        Pass output_base_dir to override.
    allowed_perturbations : set or None
        If provided, only keep cells with these perturbations (+ control)
    save_datasets_for : list of int or None
        Which HVG thresholds to save h5ad files for. Default [5000].
    """
    if output_base_dir is None:
        output_base_dir = str(config.DATASET2_DIR)

    print(f"{'='*60}")
    print(f"Preprocessing dataset: {dataset_name}")
    print(f"Input file: {input_file}")
    print(f"Output dir: {output_base_dir}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(input_file)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Check for perturbation column - create from 'gene' or 'gene_target' if needed
    if 'perturbation' not in adata.obs.columns:
        if 'gene' in adata.obs.columns:
            print("Creating adata.obs['perturbation'] from adata.obs['gene']")
            adata.obs['perturbation'] = adata.obs['gene'].copy()
        elif 'gene_target' in adata.obs.columns:
            print("Creating adata.obs['perturbation'] from adata.obs['gene_target']")
            adata.obs['perturbation'] = adata.obs['gene_target'].copy()
        else:
            raise ValueError(
                "ERROR: None of adata.obs['perturbation'], adata.obs['gene'], or adata.obs['gene_target'] columns found!\n"
                "Please ensure your dataset has one of these columns."
            )

    print(f"Perturbations found: {adata.obs['perturbation'].nunique()}")
    print(f"Perturbation counts:\n{adata.obs['perturbation'].value_counts().head()}\n")

    # Preprocess
    adata = preData_multiHVG(adata,
                            domaxNumsPerturb=domaxNumsPerturb,
                            domaxNumsControl=domaxNumsControl,
                            minNums=minNums,
                            allowed_perturbations=allowed_perturbations)

    # Save
    print(f"\n{'='*60}")
    print("Saving processed datasets...")
    print(f"{'='*60}\n")
    save_hvg_subsets(adata, dataset_name, output_base_dir, save_datasets_for=save_datasets_for)

    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_base_dir}/{dataset_name}/")
    print("Available HVG versions: hvg100/, hvg1000/, hvg2000/, hvg5000/")

    return adata
