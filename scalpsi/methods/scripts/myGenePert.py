import sys
sys.path.append('/home/software/GenePert')
import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from collections import OrderedDict
from itertools import chain
import importlib
import matplotlib.pyplot as plt
import pickle, sklearn, umap
# Reload the module
import utils # type: ignore
import GenePertExperiment  #type: ignore
importlib.reload(utils)
# Reload the module
importlib.reload(GenePertExperiment)
from utils import get_best_overall_mse_corr, run_experiments_with_embeddings, plot_mse_corr_comparison, compare_embedding_correlations #type: ignore

import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.model_selection import KFold
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import split_utils  # NEW: Use JSON-based splits



embedding_path =  "/home/software/GenePert/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"

def populate_dicts(adata_subset, mean_dict):
    """Populate mean dictionary from adata subset. Works with both 'condition' and 'perturbation' columns."""
    # Prefer 'perturbation' since JSON splits use raw perturbation names (e.g. "TP53")
    # 'condition' column has GEARS-style names (e.g. "TP53+ctrl") which causes spurious missing embedding warnings
    if 'perturbation' in adata_subset.obs.columns:
        condition_col = 'perturbation'
    elif 'condition' in adata_subset.obs.columns:
        condition_col = 'condition'
    else:
        condition_col = 'perturbation'

    for condition in adata_subset.obs[condition_col].unique():
        condition_mask = adata_subset.obs[condition_col] == condition
        condition_data = adata_subset[condition_mask].X
        # No need to clean_condition since perturbations are already clean
        mean_dict[condition] = np.mean(condition_data, axis=0)

def doLinearModel(DataSet, split_index=0, suffix='',
                  data_dir='preprocessed', split_dir='data/splits'):
    """
    Train GenePert Ridge model with JSON-based train/val/test splits.

    Args:
        DataSet: Dataset name
        split_index: Split index 0-4 (replaces old seed parameter)
        suffix: Suffix for output directory (e.g. '_hammerscalpel')
        data_dir: Base directory containing preprocessed datasets
        split_dir: Directory containing split JSON files
    """
    dirName = '{}/{}/hvg5000/GenePert{}'.format(data_dir, DataSet, suffix)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    dataset_path = "{}/{}/hvg5000/filter_hvg5000_logNor.h5ad".format(data_dir, DataSet)

    # Load dataset directly first to get perturbations for split loading
    adata_raw = sc.read_h5ad(dataset_path)

    # Load split from JSON using raw perturbations
    available_perturbations = adata_raw.obs['perturbation'].unique().tolist()
    train_perts, val_perts, test_perts, stats = split_utils.get_split_perturbations(
        split_index, available_perturbations, split_dir=split_dir, verbose=True
    )

    # Assert split correctness
    split_utils.assert_split_correctness(train_perts, val_perts, test_perts, split_index, split_dir=split_dir)

    # Now load through GenePertExperiment
    experiment = GenePertExperiment.GenePertExperiment(embeddings=None)
    experiment.load_dataset(dataset_path)
    with open(embedding_path, "rb") as fp:
        embeddings = pickle.load(fp)
    experiment.embeddings = embeddings

    # Determine which column to use for filtering
    # Prefer 'perturbation' since JSON splits use raw perturbation names (e.g. "TP53")
    # 'condition' column has GEARS-style names (e.g. "TP53+ctrl") which won't match
    if 'perturbation' in experiment.adata.obs.columns:
        condition_col = 'perturbation'
    elif 'condition' in experiment.adata.obs.columns:
        condition_col = 'condition'
    else:
        raise ValueError("Neither 'perturbation' nor 'condition' column found in adata")

    print(f"\nUsing column '{condition_col}' from GenePert experiment")
    print(f"Data splits:")
    print(f"  Train: {len(train_perts)} perturbations")
    print(f"  Val:   {len(val_perts)} perturbations")
    print(f"  Test:  {len(test_perts)} perturbations")

    embedding_size = len(next(iter(experiment.embeddings.values())))
    X_train, y_train, X_test, y_test = [], [], [], []

    # Filter by train and test splits (train only, no val)
    train_mask = experiment.adata.obs[condition_col].isin(train_perts)
    test_mask = experiment.adata.obs[condition_col].isin(test_perts)
    adata_train = experiment.adata[train_mask]
    adata_test = experiment.adata[test_mask]

    mean_dict_train, mean_dict_test = {}, {}
    populate_dicts(adata_train, mean_dict_train)
    populate_dicts(adata_test, mean_dict_test)
    train_gene_name_X_map = experiment.populate_X_y(mean_dict_train, X_train, y_train, embedding_size)
    test_gene_name_X_map = experiment.populate_X_y(mean_dict_test, X_test, y_test, embedding_size)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Train Ridge model
    ridge_model = Ridge(alpha=1, random_state=42)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)

    # Save results
    result = pd.DataFrame(y_pred, columns=experiment.adata.var_names, index=mean_dict_test.keys())
    dirOut = 'savedModels_split{}'.format(split_index)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    result.to_csv('{}/pred.tsv'.format(dirOut), sep='\t')
    print(f"\n✓ Results saved to {dirOut}/pred.tsv")


def generateExp(cellNum, means, std):
    expression_matrix = np.array([
    np.random.normal(loc=means[i], scale=std[i], size=cellNum) 
    for i in range(len(means))]).T
    return expression_matrix


### 根据预测的生成表达量
def generateH5ad(DataSet, split_index=0, data_dir='preprocessed'):
    """Generate H5AD file from predictions (updated for split_index)."""
    dirName = '{}/{}/hvg5000/GenePert'.format(data_dir, DataSet)
    os.chdir(dirName)
    filein = 'savedModels_split{}/pred.tsv'.format(split_index)
    exp = pd.read_csv(filein, sep='\t', index_col=0)
    # NOTE: This depends on GEARS results - may need updating
    filein = '../GEARS/savedModels_split{}/result.h5ad'.format(split_index)
    adata = sc.read_h5ad(filein)
    expGene = np.intersect1d(adata.var_names, exp.columns)
    pertGenes = np.intersect1d(adata.obs['perturbation'].unique(), exp.index)
    adata = adata[:, expGene]; exp = exp.loc[:, expGene]

    control_exp = adata[adata.obs['perturbation'] == 'control'].to_df()
    control_std = list(np.std(control_exp))
    control_std = [i if not np.isnan(i) else 0 for i in control_std]
    for pertGene in pertGenes:
        cellNum = adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].shape[0]
        means = list(exp.loc[pertGene, ])
        expression_matrix = generateExp(cellNum, means, control_std)
        adata[(adata.obs['perturbation'] == pertGene) & (adata.obs['Expcategory']=='imputed')].X = expression_matrix
    adata.write('savedModels_split{}/result.h5ad'.format(split_index))



### conda activate gears

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train GenePert with JSON splits')
    parser.add_argument('--split-index', type=int, default=0, choices=list(range(10)),
                        help='Split index (0-4) to use from JSON files')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to run')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for output directory (e.g. _hammerscalpel)')
    parser.add_argument('--data-dir', type=str, default='preprocessed',
                        help='Base directory containing preprocessed datasets')
    parser.add_argument('--split-dir', type=str, default='data/splits',
                        help='Directory containing split JSON files')
    args = parser.parse_args()

    print(f'\n{"="*70}')
    print(f'myGenePert.py - Using split_index={args.split_index} (JSON-based splits)')
    if args.suffix:
        print(f'Output suffix: {args.suffix}')
    print(f'{"="*70}\n')

    print(f"Processing {args.dataset}")
    doLinearModel(args.dataset, args.split_index, suffix=args.suffix,
                  data_dir=args.data_dir, split_dir=args.split_dir)