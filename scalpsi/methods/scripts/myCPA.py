import os
import scanpy as sc, pandas as pd, numpy as np, anndata as ad
import anndata
import pickle, torch, cpa, joblib
import torch.nn as nn
import split_utils
sc.settings.verbosity = 3

# Monkey-patch knn_purity to handle empty inputs during validation
import cpa._metrics as _cpa_metrics
_original_knn_purity = _cpa_metrics.knn_purity
def _safe_knn_purity(data, labels, n_neighbors=30):
    if data.shape[0] == 0:
        return 0.0
    n_neighbors = min(n_neighbors, data.shape[0] - 1)
    if n_neighbors < 1:
        return 0.0
    return _original_knn_purity(data, labels, n_neighbors=n_neighbors)
_cpa_metrics.knn_purity = _safe_knn_purity
import cpa._module as _cpa_module
_cpa_module.knn_purity = _safe_knn_purity


def getModelParameter():
    ae_hparams = {
    "n_latent": 128,
    "recon_loss": "gauss",
    "doser_type": "linear",
    "n_hidden_encoder": 512,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 256,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.2,
    "variational": False,
    "seed": 1117,
}

    trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 100,
    "n_epochs_mixup_warmup": 10,
    "mixup_alpha": 0.2,
    "adv_steps": 2,
    "n_hidden_adv": 128,
    "n_layers_adv": 2,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.2,
    "reg_adv": 50.0,
    "pen_adv": 1.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": False,
    "gradient_clip_value": 5.0,
    "step_size_lr": 45,
}
    return ae_hparams,  trainer_params


def runCPA_genetic(DataSet, split_index=0, suffix='',
                   data_dir='preprocessed', split_dir='data/splits'):
    dataset_path = '{}/{}/hvg5000/filter_hvg5000_logNor.h5ad'.format(data_dir, DataSet)
    adata = sc.read_h5ad(dataset_path)

    dirName = '{}/{}/hvg5000/CPA{}'.format(data_dir, DataSet, suffix)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    modeldir = 'savedModels_split{}'.format(split_index)
    modeldir_pt = '{}/model.pt'.format(modeldir)
    if os.path.isfile(modeldir_pt): return  # Skip if already trained

    # Load splits from JSON
    available_perturbations = adata.obs['perturbation'].unique().tolist()
    train_perts, val_perts, test_perts, stats = split_utils.get_split_perturbations(
        split_index, available_perturbations, split_dir=split_dir, verbose=True
    )
    split_utils.assert_split_correctness(train_perts, val_perts, test_perts, split_index, split_dir=split_dir)

    # Create 'split' column for CPA based on perturbation names
    split_map = {}
    for p in train_perts:
        split_map[p] = 'train'
    for p in val_perts:
        split_map[p] = 'val'
    for p in test_perts:
        split_map[p] = 'test'
    adata.obs['split'] = adata.obs['perturbation'].map(split_map)
    # Control cells: assign to train (CPA handles them via control_group)
    adata.obs.loc[adata.obs['perturbation'] == 'control', 'split'] = 'train'

    cpa.CPA.setup_anndata(adata,
                    perturbation_key='perturbation',
                    control_group='control',
                    dosage_key = None,
                    batch_key = None,
                    is_count_data= False,
                    max_comb_len=2,
                    )

    if not os.path.isdir(modeldir): os.makedirs(modeldir)
    adata.write_h5ad('{}/cpa.h5ad'.format(modeldir))

    # Load gene embeddings (inside container at /home/project/...)
    data_ge = joblib.load("/home/project/GW_PerturbSeq/geneEmbedding/scGPT.pkl")

    gene_list = list(data_ge.keys())
    gene_embeddings = np.array(list(data_ge.values()))
    gene_embeddings = np.concatenate((gene_embeddings, np.random.rand(1, gene_embeddings.shape[1])), 0)
    embeddings = anndata.AnnData(X=gene_embeddings)
    embeddings.obs.index = gene_list+['control']
    mean_embedding = embeddings.X.mean(axis=0, keepdims=True)
    perturb_genes = list(cpa.CPA.pert_encoder.keys())
    perturb_genes.remove('<PAD>')
    perturb_X = []
    valid_genes = []
    for g in perturb_genes:
        if g in embeddings.obs.index:
            perturb_X.append(embeddings[g].X[0])
        else:
            perturb_X.append(mean_embedding[0])
        valid_genes.append(g)
    perturb_X = np.stack(perturb_X, axis=0)
    embeddings = anndata.AnnData(X=perturb_X)
    embeddings.obs.index = valid_genes

    GENE_embeddings = nn.Embedding(len(perturb_genes)+1, embeddings.shape[1], padding_idx = 0)
    pad_X = np.zeros(shape=(1, embeddings.shape[1]))
    X = np.concatenate((pad_X, embeddings.X), 0)
    GENE_embeddings.weight.data.copy_(torch.tensor(X))
    GENE_embeddings.weight.requires_grad = False

    ae_hparams,  trainer_params = getModelParameter()
    model = cpa.CPA(adata=adata,
                use_rdkit_embeddings=True,
                gene_embeddings = GENE_embeddings,
                split_key='split',
                train_split='train',
                valid_split='val',
                test_split='test',
                **ae_hparams,
               )

    model.train(max_epochs=max_epoch,
            use_gpu=True,
            batch_size=1024,
            plan_kwargs=trainer_params,
            early_stopping_patience=5,
            check_val_every_n_epoch=5,
            save_path= modeldir,
           )


def getPredict(DataSet, split_index=0, suffix='', data_dir='preprocessed'):
    dirName = '{}/{}/hvg5000/CPA{}'.format(data_dir, DataSet, suffix)
    os.chdir(dirName)

    modeldir = 'savedModels_split{}'.format(split_index)
    adata = sc.read_h5ad('{}/cpa.h5ad'.format(modeldir))
    model = cpa.CPA.load(modeldir, adata = adata, use_gpu = True)

    a = adata[adata.obs['split'] == 'test'].shape[0]
    tmp = adata[adata.obs['perturbation'] == 'control']
    tmp = tmp.to_df().sample(n=a, random_state=42, replace=True)
    adata[adata.obs['split'] == 'test'].X = tmp.values

    model.predict(adata, batch_size=2048)
    adata_pred = adata[adata.obs['split'] == 'test'].copy()
    adata_pred.X = adata_pred.obsm['CPA_pred']
    adata_pred.obs['Expcategory'] = 'imputed'

    adata_ctrl = adata[adata.obs['perturbation'] == 'control']
    adata_ctrl.obs['Expcategory'] = 'control'

    adata_stimulated = adata[adata.obs['split'] == 'test'].copy()
    adata_stimulated.obs['Expcategory'] = 'stimulated'

    adata_fi = ad.concat([adata_ctrl, adata_pred, adata_stimulated])
    adata_fi.write('{}/result.h5ad'.format(modeldir))
    print(f"\nResults saved to {modeldir}/result.h5ad")


max_epoch = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### conda activate cpa

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train CPA with JSON splits')
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
    print(f'myCPA.py - Using split_index={args.split_index} (JSON-based splits)')
    if args.suffix:
        print(f'Output suffix: {args.suffix}')
    print(f'{"="*70}\n')

    print(f"Processing {args.dataset}")
    runCPA_genetic(args.dataset, args.split_index, suffix=args.suffix,
                   data_dir=args.data_dir, split_dir=args.split_dir)
    getPredict(args.dataset, args.split_index, suffix=args.suffix, data_dir=args.data_dir)
