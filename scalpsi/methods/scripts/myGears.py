#coding:utf-8
import sys, os
from scipy import sparse
import scanpy as sc
from gears import PertData, GEARS
from gears.inference import evaluate, compute_metrics
import anndata as ad
import torch
import shutil
from collections import OrderedDict
import gc
import split_utils

sc.settings.verbosity = 3


def filter_perts_by_genes(perts, gene_set):
    """Filter out perturbations whose genes are not in the gene set."""
    filtered = []
    removed = []
    for p in perts:
        genes = p.split('+')
        if all(g in gene_set or g == 'control' for g in genes):
            filtered.append(p)
        else:
            removed.append(p)
    return filtered, removed


def trainModel(adata, issplit=False, split_index=0, original_adata=None, split_dir='data/splits'):
    pert_data = PertData('./data')
    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        pert_data.new_data_process(dataset_name='train', adata=adata)
    pert_data.load(data_path='./data/train')

    pert_data.prepare_split(split='simulation', seed=split_index, train_gene_set_size=.8)

    # Override the splits with our JSON-based splits
    if original_adata is not None:
        available_perturbations = original_adata.obs['perturbation'].unique().tolist()
        train_perts, val_perts, test_perts, stats = split_utils.get_split_perturbations(
            split_index, available_perturbations, split_dir=split_dir, verbose=True
        )
        split_utils.assert_split_correctness(train_perts, val_perts, test_perts, split_index, split_dir=split_dir)

        def to_gears_condition(pert):
            if '+' in pert:
                return '+'.join(sorted(pert.split('+')))
            else:
                return f"{pert}+ctrl"

        train_conditions = [to_gears_condition(p) for p in train_perts]
        val_conditions = [to_gears_condition(p) for p in val_perts]
        test_conditions = [to_gears_condition(p) for p in test_perts]

        valid_conditions = set(pert_data.adata.obs['condition'].unique())
        def filter_conditions(conditions, split_name):
            kept = [c for c in conditions if c in valid_conditions]
            removed = [c for c in conditions if c not in valid_conditions]
            if removed:
                print(f"  Filtered {len(removed)} conditions from {split_name}: {removed}")
            return kept

        train_conditions = filter_conditions(train_conditions, 'train')
        val_conditions = filter_conditions(val_conditions, 'val')
        test_conditions = filter_conditions(test_conditions, 'test')

        pert_data.set2conditions = {
            'train': train_conditions,
            'val': val_conditions,
            'test': test_conditions
        }

        print(f"\n{'='*70}")
        print(f"GEARS: Overridden splits with JSON split{split_index}")
        print(f"{'='*70}")
        print(f"  Train: {len(train_conditions)} conditions")
        print(f"  Val:   {len(val_conditions)} conditions")
        print(f"  Test:  {len(test_conditions)} conditions")
        print(f"{'='*70}\n")

    if issplit: return

    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    gears_model = GEARS(pert_data, device=device)
    gears_model.model_initialize(hidden_size=64)
    gears_model.train(epochs=5)
    gears_model.save_model('savedModels_split{}'.format(split_index))
    return gears_model


def doGearsFormat(adata):
    def fun1(x):
        if x == 'control': return 'ctrl'
        elif '+' in x:
            genes = x.split('+')
            return genes[0] + '+' + genes[1]
        else: return x + '+' + 'ctrl'
    adata.obs['cell_type'] = 'K562'
    adata.obs['condition'] = adata.obs['perturbation'].apply(lambda x: fun1(x))
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var_names
    if not sparse.issparse(adata.X): adata.X = sparse.csr_matrix(adata.X)
    return adata


def runGears(DataSet, issplit=False, redo=False, split_index=0, suffix='',
             data_dir='preprocessed', split_dir='data/splits'):
    dirName = '{}/{}/hvg5000/GEARS{}'.format(data_dir, DataSet, suffix)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)
    if redo and os.path.isdir('data/train'):
        shutil.rmtree('data/train')

    if os.path.isfile('savedModels_split{}/model.pt'.format(split_index)): return

    data_path = f'{data_dir}/{DataSet}/hvg5000/filter_hvg5000_logNor.h5ad'
    adata_original = sc.read_h5ad(data_path)

    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        adata = adata_original.copy()
        adata.uns['log1p'] = {}; adata.uns['log1p']["base"] = None
        adata = doGearsFormat(adata)
        trainModel(adata, issplit=issplit, split_index=split_index,
                   original_adata=adata_original, split_dir=split_dir)
        del adata
    else:
        trainModel(adata='', issplit=issplit, split_index=split_index,
                   original_adata=adata_original, split_dir=split_dir)
    del adata_original
    gc.collect()


def loadModel(dirName, split_index, original_adata, split_dir='data/splits'):
    os.chdir(dirName)
    pert_data = PertData('./data')
    pert_data.load(data_path='./data/train')

    pert_data.prepare_split(split='simulation', seed=split_index, train_gene_set_size=.8)

    available_perturbations = original_adata.obs['perturbation'].unique().tolist()
    train_perts, val_perts, test_perts, stats = split_utils.get_split_perturbations(
        split_index, available_perturbations, split_dir=split_dir, verbose=False
    )

    def to_gears_condition(pert):
        if '+' in pert:
            return '+'.join(sorted(pert.split('+')))
        else:
            return f"{pert}+ctrl"

    train_conditions = [to_gears_condition(p) for p in train_perts]
    val_conditions = [to_gears_condition(p) for p in val_perts]
    test_conditions = [to_gears_condition(p) for p in test_perts]

    valid_conditions = set(pert_data.adata.obs['condition'].unique())
    train_conditions = [c for c in train_conditions if c in valid_conditions]
    val_conditions = [c for c in val_conditions if c in valid_conditions]
    test_conditions = [c for c in test_conditions if c in valid_conditions]

    pert_data.set2conditions = {
        'train': train_conditions,
        'val': val_conditions,
        'test': test_conditions
    }

    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    gears_model = GEARS(pert_data, device=device)
    gears_model.load_pretrained('savedModels_split{}'.format(split_index))
    return gears_model


def remove_duplicates_and_preserve_order(input_list):
    deduplicated_dict = OrderedDict.fromkeys(input_list)
    return list(deduplicated_dict.keys())


def getPredict(DataSet, split_index, suffix='', data_dir='preprocessed', split_dir='data/splits'):
    dirName = '{}/{}/hvg5000/GEARS{}'.format(data_dir, DataSet, suffix)
    os.chdir(dirName)

    data_path = f'{data_dir}/{DataSet}/hvg5000/filter_hvg5000_logNor.h5ad'
    adata_original = sc.read_h5ad(data_path)

    gears_model = loadModel(dirName, split_index, adata_original, split_dir=split_dir)
    del adata_original
    gc.collect()

    adata = gears_model.adata
    test_loader = gears_model.dataloader['test_loader']
    test_res = evaluate(test_loader, gears_model.best_model, gears_model.config['uncertainty'], gears_model.device)
    pert_cats = remove_duplicates_and_preserve_order(test_res['pert_cat'])

    adata_list = [adata[adata.obs['condition'] == pert_cat].copy() for pert_cat in pert_cats]
    adata2 = ad.concat(adata_list)
    del adata_list
    adata2.obs['Expcategory'] = 'stimulated'

    adata_list = [adata[adata.obs['condition'] == pert_cat].copy() for pert_cat in pert_cats]
    adata1 = ad.concat(adata_list)
    del adata_list
    adata1.X = test_res['pred']
    adata1.obs['Expcategory'] = 'imputed'

    del test_res
    adata_ctrl = gears_model.ctrl_adata
    adata_ctrl.obs['Expcategory'] = 'control'
    adata_fi = ad.concat([adata1, adata2, adata_ctrl])
    del adata1, adata2
    gc.collect()
    adata_fi.write('savedModels_split{}/result.h5ad'.format(split_index))
    print(f"\nResults saved to savedModels_split{split_index}/result.h5ad")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### conda activate gears 0.1.0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train GEARS with JSON splits')
    parser.add_argument('--split-index', type=int, default=0, choices=list(range(10)),
                        help='Split index (0-4) to use from JSON files')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to run')
    parser.add_argument('--issplit', action='store_true',
                        help='Only do split preparation without training')
    parser.add_argument('--redo', action='store_true',
                        help='Remove existing data and reprocess')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for output directory (e.g. _hammerscalpel)')
    parser.add_argument('--data-dir', type=str, default='preprocessed',
                        help='Base directory containing preprocessed datasets')
    parser.add_argument('--split-dir', type=str, default='data/splits',
                        help='Directory containing split JSON files')
    args = parser.parse_args()

    print(f'\n{"="*70}')
    print(f'myGears.py - Using split_index={args.split_index} (JSON-based splits)')
    if args.suffix:
        print(f'Output suffix: {args.suffix}')
    print(f'{"="*70}\n')

    print(f"Processing {args.dataset}")
    runGears(args.dataset, issplit=args.issplit, redo=args.redo, split_index=args.split_index,
             suffix=args.suffix, data_dir=args.data_dir, split_dir=args.split_dir)
    if not args.issplit:
        getPredict(args.dataset, args.split_index, suffix=args.suffix,
                   data_dir=args.data_dir, split_dir=args.split_dir)
