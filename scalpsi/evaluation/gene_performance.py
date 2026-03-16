"""
Gene-level performance evaluation on benchmark results.

Computes per-(perturbation, gene) metrics and gene-level aggregates.
Filters to top N DEGs per perturbation (default 2000). Outputs parquet.

Outputs two files:
  1. Gene-detail parquet: one row per (dataset, method, split, perturbation, gene)
  2. Gene-aggregate parquet: one row per (dataset, method, split, gene) with
     correlations across perturbations

Set SCALPSI_DATA_DIR to point to your DataSet2 directory.
"""

import os, sys, warnings, pickle

os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse, stats
from tqdm import tqdm
warnings.filterwarnings('ignore')

try:
    import pyarrow  # noqa: F401
except ImportError:
    try:
        import fastparquet  # noqa: F401
    except ImportError:
        raise ImportError("No parquet engine found. Install pyarrow: pip install pyarrow")

from scalpsi import config


def checkNan(adata, condition_column='perturbation', control_tag='control'):
    adata1 = adata.copy()
    if sparse.issparse(adata.X):
        adata1.X = adata.X.toarray()
    nan_rows = np.where(np.isnan(adata1.X).any(axis=1))[0]
    if len(nan_rows) >= 1:
        a = adata1[adata1.obs[condition_column] == control_tag].X.mean(axis=0)
        a = a.reshape([1, -1])
        b = np.tile(a, [len(nan_rows), 1])
        adata1[nan_rows].X = b
    return adata1


def getDEG(deg_dict, perturb, numDEG):
    DegList = list(deg_dict[perturb].index[:numDEG])
    return DegList


def build_adata_from_pred_tsv(pred_path, gt_h5ad_path, condition_column='perturbation', control_tag='control', gt_adata=None):
    """Build a result.h5ad-like anndata from a pred.tsv and ground truth h5ad."""
    pred_df = pd.read_csv(pred_path, sep='\t', index_col=0)
    if gt_adata is None:
        gt_adata = sc.read_h5ad(gt_h5ad_path)

    if sparse.issparse(gt_adata.X):
        gt_adata.X = gt_adata.X.toarray()

    test_perturbations = pred_df.index.tolist()
    gene_names = pred_df.columns.tolist()

    ctrl_adata = gt_adata[gt_adata.obs[condition_column] == control_tag, gene_names].copy()
    ctrl_adata.obs = ctrl_adata.obs[[condition_column]].copy()
    ctrl_adata.obs['Expcategory'] = control_tag

    parts = [ctrl_adata]

    for perturb in test_perturbations:
        stim_cells = gt_adata[gt_adata.obs[condition_column] == perturb, gene_names].copy()
        if stim_cells.shape[0] == 0:
            continue
        stim_cells.obs = stim_cells.obs[[condition_column]].copy()
        stim_cells.obs['Expcategory'] = 'stimulated'

        n_cells = stim_cells.shape[0]
        pred_mean = pred_df.loc[perturb].values.astype(np.float32)
        imp_X = np.tile(pred_mean, (n_cells, 1))
        imp_obs = pd.DataFrame({
            condition_column: [perturb] * n_cells,
            'Expcategory': ['imputed'] * n_cells,
        })
        imp_adata = ad.AnnData(X=imp_X, obs=imp_obs)
        imp_adata.var_names = gene_names

        parts.extend([stim_cells, imp_adata])

    combined = ad.concat(parts, join='inner')
    combined.obs_names_make_unique()
    return combined


def get_gt_h5ad_for_method(base_dir, dataset_name, hvg, method):
    """Resolve ground-truth h5ad path for a given method.

    Convention:
    - methods containing:
      * 'k562matchfilterfirst' -> filter_hvg{hvg}K562matchfilterfirst_logNor.h5ad
      * 'k562match'            -> filter_hvg{hvg}K562Match_logNor.h5ad
      * 'nomatch'              -> filter_hvg{hvg}noMatch_logNor.h5ad
    - otherwise:
      * methods containing 'full' -> filter_hvg{hvg}full_logNor.h5ad
      * else                      -> filter_hvg{hvg}_logNor.h5ad
    """
    hvg_dir = os.path.join(base_dir, dataset_name, f'hvg{hvg}')
    method_l = str(method).lower()

    if 'k562matchfilterfirst' in method_l:
        return os.path.join(hvg_dir, f'filter_hvg{hvg}K562matchfilterfirst_logNor.h5ad')
    elif 'k562match' in method_l:
        return os.path.join(hvg_dir, f'filter_hvg{hvg}K562Match_logNor.h5ad')
    elif 'nomatch' in method_l:
        return os.path.join(hvg_dir, f'filter_hvg{hvg}noMatch_logNor.h5ad')
    elif 'full' in method_l:
        candidates = [
            os.path.join(hvg_dir, f'filter_hvg{hvg}full_logNor.h5ad'),
            os.path.join(hvg_dir, f'filter_hvg{hvg}_logNor.h5ad'),
        ]
    else:
        candidates = [
            os.path.join(hvg_dir, f'filter_hvg{hvg}_logNor.h5ad'),
            os.path.join(hvg_dir, f'filter_hvg{hvg}full_logNor.h5ad'),
        ]

    primary = candidates[0]
    for p in candidates:
        if os.path.isfile(p):
            if p != primary:
                print(f"  WARNING: preferred GT file missing for method={method}; using fallback: {p}")
            return p
    return primary


def compute_gene_metrics(adata, deg_dict, dataset_name, method, split,
                         numDEG=2000, use_all_genes=False,
                         condition_column='perturbation', control_tag='control'):
    """Compute per-(perturbation, gene) metrics.

    Returns a DataFrame with one row per (perturbation, gene). By default,
    includes top `numDEG` DEGs per perturbation from `deg_dict`.
    If `use_all_genes=True`, evaluates all genes available in `adata`.
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()

    adata = checkNan(adata, condition_column, control_tag)

    control_list = ['control', 'MCF7_control_1.0', 'A549_control_1.0', 'K562_control_1.0']
    perturbations = adata.obs[condition_column].unique()
    perturbations = [p for p in perturbations if p not in control_list]

    ctrl_cells = adata[adata.obs['Expcategory'] == control_tag]
    ctrl_mean_full = np.array(ctrl_cells.X.mean(axis=0)).ravel()
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    rows = []
    for perturb in tqdm(perturbations, desc=f"    {method} split={split}"):
        if use_all_genes:
            eval_genes = gene_names
            eval_indices = np.arange(len(gene_names), dtype=int)
        else:
            if perturb not in deg_dict:
                continue
            eval_genes = getDEG(deg_dict, perturb, numDEG)
            eval_genes = [g for g in eval_genes if g in gene_to_idx]
            if len(eval_genes) == 0:
                continue
            eval_indices = [gene_to_idx[g] for g in eval_genes]

        stim = adata[(adata.obs[condition_column] == perturb) & (adata.obs['Expcategory'] == 'stimulated')]
        imp = adata[(adata.obs[condition_column] == perturb) & (adata.obs['Expcategory'] == 'imputed')]

        if stim.shape[0] == 0 or imp.shape[0] == 0:
            continue

        stim_X = np.array(stim.X)[:, eval_indices]
        imp_X = np.array(imp.X)[:, eval_indices]
        ctrl_mean = ctrl_mean_full[eval_indices]

        mean_true = stim_X.mean(axis=0)
        mean_pred = imp_X.mean(axis=0)
        var_true = stim_X.var(axis=0)
        var_pred = imp_X.var(axis=0)
        mean_true_delta = mean_true - ctrl_mean
        mean_pred_delta = mean_pred - ctrl_mean
        abs_error = np.abs(mean_pred - mean_true)
        squared_error = (mean_pred - mean_true) ** 2
        abs_error_delta = np.abs(mean_pred_delta - mean_true_delta)
        squared_error_delta = (mean_pred_delta - mean_true_delta) ** 2

        for j, gene in enumerate(eval_genes):
            rows.append({
                'dataset': dataset_name,
                'method': method,
                'split': split,
                'perturbation': perturb,
                'gene': gene,
                'mean_pred': round(mean_pred[j], 6),
                'mean_true': round(mean_true[j], 6),
                'mean_pred_delta': round(mean_pred_delta[j], 6),
                'mean_true_delta': round(mean_true_delta[j], 6),
                'var_pred': round(var_pred[j], 6),
                'var_true': round(var_true[j], 6),
                'abs_error': round(abs_error[j], 6),
                'squared_error': round(squared_error[j], 6),
                'abs_error_delta': round(abs_error_delta[j], 6),
                'squared_error_delta': round(squared_error_delta[j], 6),
                'deg_rank': j + 1,
                'n_stim_cells': stim.shape[0],
                'n_imp_cells': imp.shape[0],
            })

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_gene_aggregates(detail_df):
    """Compute per-gene aggregated metrics across perturbations.

    For each (dataset, method, split, gene), computes Pearson and Spearman
    correlations between predicted and true means across all perturbations.
    """
    agg_rows = []
    grouped = detail_df.groupby(['dataset', 'method', 'split', 'gene'])
    for (dataset, method, split, gene), group in tqdm(grouped, total=len(grouped), desc="    Aggregating genes"):
        n_perts = len(group)
        if n_perts < 3:
            continue

        pcc_raw, pcc_raw_pval = stats.pearsonr(group['mean_pred'], group['mean_true'])
        scc_raw, scc_raw_pval = stats.spearmanr(group['mean_pred'], group['mean_true'])
        pcc_delta, pcc_delta_pval = stats.pearsonr(group['mean_pred_delta'], group['mean_true_delta'])
        scc_delta, scc_delta_pval = stats.spearmanr(group['mean_pred_delta'], group['mean_true_delta'])

        agg_rows.append({
            'dataset': dataset,
            'method': method,
            'split': split,
            'gene': gene,
            'n_perturbations': n_perts,
            'pearson_raw': round(pcc_raw, 6),
            'pearson_raw_pval': pcc_raw_pval,
            'spearman_raw': round(scc_raw, 6),
            'spearman_raw_pval': scc_raw_pval,
            'pearson_delta': round(pcc_delta, 6),
            'pearson_delta_pval': pcc_delta_pval,
            'spearman_delta': round(scc_delta, 6),
            'spearman_delta_pval': scc_delta_pval,
            'mse_raw': round(group['squared_error'].mean(), 6),
            'mae_raw': round(group['abs_error'].mean(), 6),
            'mse_delta': round(group['squared_error_delta'].mean(), 6),
            'mae_delta': round(group['abs_error_delta'].mean(), 6),
            'mean_var_pred': round(group['var_pred'].mean(), 6),
            'mean_var_true': round(group['var_true'].mean(), 6),
        })

    if not agg_rows:
        return None
    return pd.DataFrame(agg_rows)


def evaluate_result(base_dir, dataset_name, hvg, method, split, deg_dict,
                    numDEG=2000, use_all_genes=False,
                    condition_column='perturbation', control_tag='control', gt_adata_cache=None):
    """Load a result and compute gene-level metrics."""
    split_dir = os.path.join(base_dir, dataset_name, f'hvg{hvg}', method, f'savedModels_split{split}')
    result_path = os.path.join(split_dir, 'result.h5ad')
    pred_path = os.path.join(split_dir, 'pred.tsv')

    if os.path.isfile(result_path):
        print(f"  Evaluating {method} split={split} (result.h5ad)...")
        adata = sc.read_h5ad(result_path)
    elif os.path.isfile(pred_path):
        print(f"  Evaluating {method} split={split} (pred.tsv)...")
        gt_h5ad = get_gt_h5ad_for_method(base_dir, dataset_name, hvg, method)
        if not os.path.isfile(gt_h5ad):
            print(f"  Skipping {method}/savedModels_split{split} - ground truth {gt_h5ad} not found")
            return None
        gt_adata = None
        if gt_adata_cache is not None:
            gt_adata = gt_adata_cache.get(gt_h5ad)
        adata = build_adata_from_pred_tsv(pred_path, gt_h5ad, condition_column, control_tag, gt_adata=gt_adata)
    else:
        print(f"  Skipping {method}/savedModels_split{split} - no result.h5ad or pred.tsv found")
        return None

    return compute_gene_metrics(adata, deg_dict, dataset_name, method, split,
                                numDEG, use_all_genes, condition_column, control_tag)


def run_gene_evaluation(datasets, hvg=1000, methods=None, splits=None,
                        base_dir=None, output_prefix=None, num_deg=2000,
                        use_all_genes=False, workers=1):
    """
    Run gene-level evaluation for the given datasets.

    Parameters
    ----------
    datasets : list of str
    hvg : int
    methods : list of str or None
    splits : list of int or None
    base_dir : str or None
        Defaults to config.DATASET2_DIR.
    output_prefix : str or None
    num_deg : int
    use_all_genes : bool
    workers : int

    Returns
    -------
    detail_df : pd.DataFrame
    agg_df : pd.DataFrame or None
    """
    if base_dir is None:
        base_dir = str(config.DATASET2_DIR)
    if splits is None:
        splits = [0, 1, 2, 3, 4]

    all_detail = []

    for dataset_name in datasets:
        print(f"\n{'---'*20}")
        print(f"Dataset: {dataset_name}")
        print(f"{'---'*20}")

        deg_dict = None
        if use_all_genes:
            print("Skipping DEG file: using all genes.\n")
        else:
            deg_file = os.path.join(base_dir, dataset_name, f'DEG_hvg{hvg}.pkl')
            if not os.path.isfile(deg_file):
                print(f"ERROR: DEG file not found: {deg_file}")
                continue
            print(f"Loading DEG file: {deg_file}")
            with open(deg_file, 'rb') as fin:
                deg_dict = pickle.load(fin)
            print(f"DEGs available for {len(deg_dict)} perturbations\n")

        hvg_dir = os.path.join(base_dir, dataset_name, f'hvg{hvg}')
        eval_methods = methods
        if eval_methods is None:
            eval_methods = [d for d in os.listdir(hvg_dir)
                            if os.path.isdir(os.path.join(hvg_dir, d)) and d != 'data']
            print(f"Auto-detected methods: {eval_methods}")
        print(f"Methods: {eval_methods}\n")

        gt_adata_cache = {}
        if workers <= 1:
            needed_gt_paths = set()
            for m in eval_methods:
                for s in splits:
                    result_path = os.path.join(hvg_dir, m, f'savedModels_split{s}', 'result.h5ad')
                    pred_path = os.path.join(hvg_dir, m, f'savedModels_split{s}', 'pred.tsv')
                    if (not os.path.isfile(result_path)) and os.path.isfile(pred_path):
                        needed_gt_paths.add(get_gt_h5ad_for_method(base_dir, dataset_name, hvg, m))

            for gt_h5ad_path in sorted(needed_gt_paths):
                if not os.path.isfile(gt_h5ad_path):
                    print(f"Ground truth not found, will skip dependent evals: {gt_h5ad_path}")
                    continue
                print(f"Pre-loading ground truth: {gt_h5ad_path}")
                gt_adata = sc.read_h5ad(gt_h5ad_path)
                if sparse.issparse(gt_adata.X):
                    gt_adata.X = gt_adata.X.toarray()
                gt_adata_cache[gt_h5ad_path] = gt_adata

        for method in eval_methods:
            for split in splits:
                result = evaluate_result(base_dir, dataset_name, hvg, method, split,
                                         deg_dict, numDEG=num_deg, use_all_genes=use_all_genes,
                                         gt_adata_cache=gt_adata_cache)
                if result is not None:
                    all_detail.append(result)

    if not all_detail:
        raise RuntimeError("No results found! Check that result.h5ad or pred.tsv files exist.")

    detail_df = pd.concat(all_detail, ignore_index=True)
    agg_df = compute_gene_aggregates(detail_df)

    datasets_str = '_'.join(datasets)
    prefix = output_prefix or os.path.join(base_dir, f'geneperf_{datasets_str}_hvg{hvg}')

    detail_file = f'{prefix}_detail.parquet'
    detail_df.to_parquet(detail_file, index=False)
    print(f"\nDetail file:    {detail_file}  ({detail_df.shape[0]} rows)")

    if agg_df is not None:
        agg_file = f'{prefix}_aggregate.parquet'
        agg_df.to_parquet(agg_file, index=False)
        print(f"Aggregate file: {agg_file}  ({agg_df.shape[0]} rows)")

    return detail_df, agg_df
