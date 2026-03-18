import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
import split_utils
warnings.filterwarnings('ignore')


def predExp_single(DataSet, split_index=1, senario='trainMean', suffix='',
                   data_dir='preprocessed', split_dir='data/splits'):
    dataset_path = '{}/{}/hvg5000/filter_hvg5000_logNor.h5ad'.format(data_dir, DataSet)
    adata = sc.read_h5ad(dataset_path)

    # Load splits from JSON
    available_perturbations = adata.obs['perturbation'].unique().tolist()
    train_perts, val_perts, test_perts, stats = split_utils.get_split_perturbations(
        split_index, available_perturbations, split_dir=split_dir, verbose=True
    )

    dirName = '{}/{}/hvg5000/{}{}'.format(data_dir, DataSet, senario, suffix)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    if senario == 'trainMean':
        adata_train = adata[adata.obs['perturbation'].isin(train_perts)]
    else:
        adata_train = adata[adata.obs['perturbation'].isin(['control'])]
    exp = adata_train.to_df()
    train_mean = exp.mean(axis=0).to_frame().T
    pred = pd.concat([train_mean] * len(test_perts))
    pred.index = test_perts

    dirOut = 'savedModels_split{}'.format(split_index)
    if not os.path.isdir(dirOut): os.makedirs(dirOut)
    pred.to_csv('{}/pred.tsv'.format(dirOut),  sep='\t')


### conda activate gears

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train linearModel_mean with JSON splits')
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
    print(f'linearModel_mean.py - Using split_index={args.split_index} (JSON-based splits)')
    if args.suffix:
        print(f'Output suffix: {args.suffix}')
    print(f'{"="*70}\n')

    for senario in ['trainMean', 'controlMean']:
        print(f"Processing {args.dataset} with {senario}")
        predExp_single(args.dataset, split_index=args.split_index, senario=senario,
                       suffix=args.suffix, data_dir=args.data_dir, split_dir=args.split_dir)
