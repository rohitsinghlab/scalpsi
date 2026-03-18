import json
import os
import sys
import time
import copy
from tqdm import tqdm
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings
import anndata as ad
import torch
from scipy import sparse
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab   #type: ignore
from torchtext._torchtext import (   #type: ignore
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
import scanpy as sc
import scgpt as scg  #type: ignore
from scgpt.model import TransformerGenerator   #type: ignore
from scgpt.loss import (   #type: ignore
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch   #type: ignore
from scgpt.tokenizer.gene_tokenizer import GeneVocab   #type: ignore
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics   #type: ignore
import split_utils


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


matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

# settings for data processing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

# settings for training
MLM = True
CLS = False
CCE = False
MVC = False
ECS = False
amp = True
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4
batch_size = 64
eval_batch_size = 64
schedule_interval = 1
early_stop = 3

# settings for the model
embsize = 512
d_hid = 512
nlayers = 12
nhead = 8
n_layers_cls = 3
dropout = 0
use_fast_transformer = True

# logging
log_interval = 100
load_model = "/home/software/scGPT/scGPT_human"


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


def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device, gene_ids
) -> Dict:
    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(float)
    results["truth"] = truth.detach().cpu().numpy().astype(float)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)

    return results


def train_scGPT(DataSet, istrain=True, split_index=0, suffix='',
                data_dir='preprocessed', split_dir='data/splits'):
    def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
        model.train()
        total_loss, total_mse = 0.0, 0.0
        start_time = time.time()

        num_batches = len(train_loader)
        for batch, batch_data in enumerate(train_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                else:
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=device
                )

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool
                )
                loss = loss_mse = criterion(output_values, target_values, masked_positions)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
                )
                total_loss = 0
                total_mse = 0
                start_time = time.time()

    dirName = '{}/{}/hvg5000/scGPT{}'.format(data_dir, DataSet, suffix)
    if not os.path.isdir(dirName): os.makedirs(dirName)
    os.chdir(dirName)

    modeloutPT = "savedModels_split{}/best_model.pt".format(split_index)
    if os.path.isfile(modeloutPT) and istrain: return
    print(DataSet)
    logger = scg.logger
    scg.utils.add_file_handler(logger, "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

    adata = sc.read_h5ad(f'{data_dir}/{DataSet}/hvg5000/filter_hvg5000_logNor.h5ad')
    adata.uns['log1p'] = {}; adata.uns['log1p']["base"] = None
    adata = doGearsFormat(adata)

    # Filter out perturbations whose genes are not in the adata gene names
    gene_names_set = set(adata.var['gene_name'].values) if 'gene_name' in adata.var.columns else set(adata.var_names)
    def _pert_genes_in_adata(condition):
        if condition == 'ctrl':
            return True
        genes = condition.replace('+ctrl', '').split('+')
        return all(g == 'ctrl' or g in gene_names_set for g in genes)
    mask = adata.obs['condition'].apply(_pert_genes_in_adata)
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"Pre-filtering: removed {n_removed} cells with perturbation genes not in adata gene names")
    adata = adata[mask].copy()

    pert_data = PertData('./data')
    if not os.path.isfile('data/train/data_pyg/cell_graphs.pkl'):
        pert_data.new_data_process(dataset_name='train', adata=adata)
    pert_data.load(data_path='./data/train')

    pert_data.prepare_split(split='simulation', seed=split_index, train_gene_set_size=.8)
    # Override with JSON splits
    adata_original = sc.read_h5ad(f'{data_dir}/{DataSet}/hvg5000/filter_hvg5000_logNor.h5ad')
    available_perturbations = adata_original.obs['perturbation'].unique().tolist()
    train_perts, val_perts, test_perts, stats = split_utils.get_split_perturbations(
        split_index, available_perturbations, split_dir=split_dir, verbose=True
    )
    split_utils.assert_split_correctness(train_perts, val_perts, test_perts, split_index, split_dir=split_dir)

    # Filter out perturbations whose genes are not in the GEARS-processed adata
    if 'gene_name' in pert_data.adata.var.columns:
        gene_set = set(pert_data.adata.var['gene_name'].values)
    else:
        gene_set = set(pert_data.adata.var_names)
    train_perts, removed_train = filter_perts_by_genes(train_perts, gene_set)
    val_perts, removed_val = filter_perts_by_genes(val_perts, gene_set)
    test_perts, removed_test = filter_perts_by_genes(test_perts, gene_set)
    all_removed = removed_train + removed_val + removed_test
    if all_removed:
        print(f"Filtered out {len(all_removed)} perturbations not in adata gene names:")
        print(f"  Train: {len(removed_train)}, Val: {len(removed_val)}, Test: {len(removed_test)}")
        print(f"  Removed: {all_removed}")

    def to_gears_condition(pert):
        if '+' in pert:
            return '+'.join(sorted(pert.split('+')))
        else:
            return f"{pert}+ctrl"

    train_conditions = [to_gears_condition(p) for p in train_perts]
    val_conditions = [to_gears_condition(p) for p in val_perts]
    test_conditions = [to_gears_condition(p) for p in test_perts]

    pert_data.set2conditions = {
        'train': train_conditions,
        'val': val_conditions,
        'test': test_conditions
    }

    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    model_dir = Path(load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)

    ntokens = len(vocab)
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )

    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }
    for k, v in pretrained_dict.items():
        logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.to(device)

    if not istrain:  return pert_data, model, gene_ids

    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val_loss = float("inf")
    best_val_corr = 0
    best_model = None
    patience = 0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train(
            model,
            train_loader,
        )

        val_res = eval_perturb(valid_loader, model, device, gene_ids)
        val_metrics = compute_perturbation_metrics(
            val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        )
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_metrics)

        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

        val_score = val_metrics["pearson"]
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {val_score:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break
        scheduler.step()
    if not os.path.isdir('savedModels_split{}'.format(split_index)): os.makedirs('savedModels_split{}'.format(split_index))
    torch.save(best_model.state_dict(), modeloutPT)


def doPredict(DataSet, split_index=0, suffix='', data_dir='preprocessed', split_dir='data/splits'):
    os.chdir('{}/{}/hvg5000/scGPT{}/'.format(data_dir, DataSet, suffix))
    resultOut = 'savedModels_split{}/result.h5ad'.format(split_index)
    filein = 'savedModels_split{}/best_model.pt'.format(split_index)
    pert_data, model, gene_ids = train_scGPT(DataSet, istrain=False, split_index=split_index,
                                              suffix=suffix, data_dir=data_dir, split_dir=split_dir)
    model.load_state_dict(torch.load(filein, map_location=device))
    test_loader = pert_data.dataloader["test_loader"]
    test_res = eval_perturb(test_loader, model, device, gene_ids)
    adata = pert_data.adata

    adata_truth = adata[adata.obs['condition'].isin(test_res['pert_cat'])]
    adata_truth.obs['Expcategory'] = 'stimulated'

    adata_pred = adata_truth.copy()
    adata_pred.X = test_res['pred']
    adata_pred.obs['Expcategory'] = 'imputed'

    adata_ctrl = adata[adata.obs['perturbation'].isin(['control'])]
    adata_ctrl.obs['Expcategory'] = 'control'

    result = ad.concat([adata_ctrl, adata_pred, adata_truth])
    result.X = result.X.toarray()
    result.write(resultOut)


epochs = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(42)

### conda activate scGPT

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train scGPT with JSON splits')
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

    print(f'{"="*70}')
    print(f'myscGPT1.py - Using split_index={args.split_index} (JSON-based splits)')
    if args.suffix:
        print(f'Output suffix: {args.suffix}')
    print(f'{"="*70}')

    train_scGPT(args.dataset, istrain=True, split_index=args.split_index, suffix=args.suffix,
                data_dir=args.data_dir, split_dir=args.split_dir)
    doPredict(args.dataset, split_index=args.split_index, suffix=args.suffix,
              data_dir=args.data_dir, split_dir=args.split_dir)
