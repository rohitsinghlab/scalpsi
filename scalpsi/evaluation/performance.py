"""
Perturbation-level performance evaluation.

Evaluates method predictions against ground truth using distance metrics
implemented with numpy and scipy (no external dependencies beyond the core stack).

Supported metrics: root_mean_squared_error, mean_absolute_error,
                   pearson_distance, spearman_distance, cosine_distance

Usage via CLI:
    scalpsi-evaluate --dataset hepg2 jurkat rpe1 --hvg 5000 --methods GEARS scGPT CPA
    scalpsi-evaluate --dataset toy --hvg 1000 --methods GEARS --splits 0 1 2

Set SCALPSI_DATA_DIR to point to your DataSet2 directory.
"""

import os, sys, warnings, pickle, gc
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse, stats
from scipy.spatial.distance import cosine as scipy_cosine
from tqdm import tqdm
warnings.filterwarnings('ignore')

from scalpsi import config


def compute_distance(imp_X, stim_X, metric):
    """
    Compute a scalar distance between the mean of imputed cells and mean of stimulated cells.

    Parameters
    ----------
    imp_X : np.ndarray, shape (n_imp, n_genes)
    stim_X : np.ndarray, shape (n_stim, n_genes)
    metric : str

    Returns
    -------
    float
    """
    mean_imp = imp_X.mean(axis=0)
    mean_stim = stim_X.mean(axis=0)

    if metric == 'root_mean_squared_error':
        return float(np.sqrt(np.mean((mean_imp - mean_stim) ** 2)))
    elif metric == 'mean_absolute_error':
        return float(np.mean(np.abs(mean_imp - mean_stim)))
    elif metric == 'pearson_distance':
        r, _ = stats.pearsonr(mean_imp, mean_stim)
        return float(1.0 - r)
    elif metric == 'spearman_distance':
        r, _ = stats.spearmanr(mean_imp, mean_stim)
        return float(1.0 - r)
    elif metric == 'cosine_distance':
        return float(scipy_cosine(mean_imp, mean_stim))
    else:
        raise ValueError(f"Unsupported metric: '{metric}'. "
                         f"Supported: root_mean_squared_error, mean_absolute_error, "
                         f"pearson_distance, spearman_distance, cosine_distance")


def checkNan(adata, condition_column='perturbation', control_tag='control'):
    adata1 = adata.copy()
    if sparse.issparse(adata1.X):
        adata1.X = np.asarray(adata1.X.todense())
    nan_rows = np.where(np.isnan(adata1.X).any(axis=1))[0]
    if len(nan_rows) >= 1:
        a = adata1[adata1.obs[condition_column] == control_tag].X.mean(axis=0)
        a = a.reshape([1, -1])
        b = np.tile(a, [len(nan_rows), 1])
        adata1[nan_rows].X = b
    return adata1


def calculateDelta(adata):
    adata_control = adata[adata.obs['Expcategory'] == 'control'].copy()
    adata_imputed = adata[adata.obs['Expcategory'] == 'imputed'].copy()
    adata_stimulated = adata[adata.obs['Expcategory'] == 'stimulated'].copy()
    control_mean = adata_control.X.mean(axis=0)
    adata_imputed.X = adata_imputed.X - control_mean
    adata_stimulated.X = adata_stimulated.X - control_mean
    return ad.concat([adata_control, adata_imputed, adata_stimulated])


def getDEG(deg_dict, perturb, numDEG):
    DegList = list(deg_dict[perturb].index[:numDEG])
    return DegList


def build_adata_from_pred_tsv(pred_path, gt_h5ad_path, condition_column='perturbation', control_tag='control', gt_adata=None):
    """Build a result.h5ad-like anndata from a pred.tsv and ground truth h5ad.

    For mean-based methods (trainMean, controlMean), pred.tsv contains one row per
    test perturbation with the predicted mean expression. We replicate this mean
    to match the number of ground truth stimulated cells per perturbation, then
    combine with control and stimulated cells with Expcategory labels.

    If gt_adata is provided, it is used directly (avoids redundant file reads).
    """
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


def calPerfor(adata, deg_dict, dataset_name, method, numDEG, split, perturb,
              condition_column, control_tag, metric):
    try:
        stim_mask = (adata.obs[condition_column] == perturb) & (adata.obs['Expcategory'] == 'stimulated')
        imp_mask  = (adata.obs[condition_column] == perturb) & (adata.obs['Expcategory'] == 'imputed')
        if stim_mask.sum() == 0 or imp_mask.sum() == 0:
            return None

        adata_sub = adata[adata.obs[condition_column].isin([control_tag, perturb])]
        DegList = getDEG(deg_dict, perturb, numDEG)
        DegList = [i for i in DegList if i in adata_sub.var_names]
        if len(DegList) == 0:
            return None
        adata_sub = adata_sub[:, DegList].copy()
        adata_sub = checkNan(adata_sub, condition_column, control_tag)

        # pearson_distance is computed on delta (expression - control mean)
        if metric == 'pearson_distance':
            adata_sub = calculateDelta(adata_sub)

        X = adata_sub.X
        if sparse.issparse(X):
            X = np.asarray(X.todense())

        imp_X  = X[(adata_sub.obs['Expcategory'] == 'imputed').values]
        stim_X = X[(adata_sub.obs['Expcategory'] == 'stimulated').values]

        perf = round(compute_distance(imp_X, stim_X, metric), 4)

        dat = pd.DataFrame({'performance': [perf], 'metric': metric})
        dat['DataSet'] = dataset_name
        dat['method'] = method
        dat['perturb'] = perturb
        dat['DEG'] = numDEG
        dat['Ncontrol']   = (adata_sub.obs['Expcategory'] == 'control').sum()
        dat['Nimputed']   = imp_X.shape[0]
        dat['Nstimulated'] = stim_X.shape[0]
        dat['split'] = split
        return dat
    except Exception as e:
        print(f"  Error evaluating {perturb} with {metric}: {e}")
        return None


def evaluate_result(base_dir, dataset_name, hvg, method, split, deg_dict, metrics,
                    condition_column='perturbation', control_tag='control', gt_adata=None):
    """Evaluate a single result.h5ad or pred.tsv file."""
    split_dir = os.path.join(base_dir, dataset_name, f'hvg{hvg}', method, f'savedModels_split{split}')
    result_path = os.path.join(split_dir, 'result.h5ad')
    pred_path = os.path.join(split_dir, 'pred.tsv')

    if os.path.isfile(result_path):
        print(f"  Evaluating {method} split={split} (result.h5ad)...")
        adata = sc.read_h5ad(result_path)
        if not sparse.issparse(adata.X):
            adata.X = sparse.csr_matrix(adata.X)
    elif os.path.isfile(pred_path):
        print(f"  Evaluating {method} split={split} (pred.tsv)...")
        gt_h5ad = os.path.join(base_dir, dataset_name, f'hvg{hvg}', f'filter_hvg{hvg}_logNor.h5ad')
        if not os.path.isfile(gt_h5ad):
            print(f"  Skipping {method}/savedModels_split{split} - ground truth {gt_h5ad} not found")
            return None
        adata = build_adata_from_pred_tsv(pred_path, gt_h5ad, condition_column, control_tag, gt_adata=gt_adata)
    else:
        print(f"  Skipping {method}/savedModels_split{split} - no result.h5ad or pred.tsv found")
        return None

    control_list = ['control', 'MCF7_control_1.0', 'A549_control_1.0', 'K562_control_1.0']
    perturbations = adata.obs[condition_column].unique()
    perturbations = [i for i in perturbations if i not in control_list]

    results_list = []
    total = len(metrics) * len(perturbations)
    pbar = tqdm(total=total, desc=f"    {method} split={split}")

    numDEG_list = [100, 1000, 2000, 5000]
    for metric in metrics:
        for perturb in perturbations:
            for numDEG in numDEG_list:
                result = calPerfor(adata, deg_dict, dataset_name, method, numDEG, split,
                                   perturb, condition_column, control_tag, metric)
                if result is not None:
                    results_list.append(result)
            gc.collect()
            pbar.update(1)

    pbar.close()

    if results_list:
        return pd.concat(results_list)
    return None


# Default metrics used in the benchmark
DEFAULT_METRICS = [
    'pearson_distance',
    'spearman_distance',
    'root_mean_squared_error',
    'mean_absolute_error',
    'cosine_distance',
]


def run_evaluation(datasets, hvg=1000, methods=None, splits=None,
                   base_dir=None, output=None, metrics=None, workers=1):
    """
    Evaluate all methods/splits for the given datasets.

    Parameters
    ----------
    datasets : list of str
        Dataset names to evaluate (must exist in base_dir).
    hvg : int
        HVG threshold used during preprocessing.
    methods : list of str or None
        Methods to evaluate. None = auto-detect from directory.
    splits : list of int or None
        Splits to evaluate. Default: [0, 1, 2, 3, 4].
    base_dir : str or None
        DataSet2 directory. Defaults to config.DATASET2_DIR.
    output : str or None
        Output file path. Default: {base_dir}/performance_{datasets}_hvg{hvg}.tsv
    metrics : list of str or None
        Metrics to compute. Default: DEFAULT_METRICS.
    workers : int
        Parallel workers (default: 1 = serial).

    Returns
    -------
    results : pd.DataFrame
    """
    if base_dir is None:
        base_dir = str(config.DATASET2_DIR)
    if splits is None:
        splits = [0, 1, 2, 3, 4]
    if metrics is None:
        metrics = DEFAULT_METRICS

    all_results = []

    for dataset_name in datasets:
        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*60}")

        deg_file = os.path.join(base_dir, dataset_name, f'DEG_hvg{hvg}.pkl')
        if not os.path.isfile(deg_file):
            print(f"ERROR: DEG file not found: {deg_file}")
            print(f"Run preprocessing first: scalpsi-preprocess --path data.h5ad --name {dataset_name}")
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

        gt_h5ad_path = os.path.join(base_dir, dataset_name, f'hvg{hvg}', f'filter_hvg{hvg}_logNor.h5ad')
        gt_adata_cached = None
        if workers <= 1 and os.path.isfile(gt_h5ad_path):
            needs_gt = any(
                not os.path.isfile(os.path.join(hvg_dir, m, f'savedModels_split{s}', 'result.h5ad'))
                and os.path.isfile(os.path.join(hvg_dir, m, f'savedModels_split{s}', 'pred.tsv'))
                for m in eval_methods for s in splits
            )
            if needs_gt:
                print(f"Pre-loading ground truth: {gt_h5ad_path}")
                gt_adata_cached = sc.read_h5ad(gt_h5ad_path)
                if sparse.issparse(gt_adata_cached.X):
                    gt_adata_cached.X = gt_adata_cached.X.toarray()

        if workers > 1:
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for method in eval_methods:
                    for split in splits:
                        future = executor.submit(
                            evaluate_result, base_dir, dataset_name, hvg,
                            method, split, deg_dict, metrics)
                        futures[future] = (method, split)
                for future in as_completed(futures):
                    method, split = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        print(f"  Error in {method} split={split}: {e}")
        else:
            for method in eval_methods:
                for split in splits:
                    result = evaluate_result(base_dir, dataset_name, hvg, method, split,
                                             deg_dict, metrics, gt_adata=gt_adata_cached)
                    if result is not None:
                        all_results.append(result)

    if not all_results:
        raise RuntimeError("No results found! Check that result.h5ad files exist.")

    results = pd.concat(all_results)
    datasets_str = '_'.join(datasets)
    output_file = output or os.path.join(base_dir, f'performance_{datasets_str}_hvg{hvg}.tsv')
    results.to_csv(output_file, sep='\t', index=False)
    print(f"\nResults saved: {output_file}")
    print(results.groupby(['DataSet', 'method', 'metric'])['performance'].mean().unstack().round(4))
    return results
