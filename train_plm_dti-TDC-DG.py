import os
import sys
import torch
import pandas as pd
import pickle as pk
import numpy as np
import typing as T
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from omegaconf import OmegaConf

BASE_DIR = "."
MODEL_BASE_DIR = f"{BASE_DIR}/best_models"
DATA_DIR = f"{BASE_DIR}/nbdata"
LOG_DIR = f"{BASE_DIR}/logs"
os.makedirs(MODEL_BASE_DIR,exist_ok=True)
os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(LOG_DIR,exist_ok=True)
sys.path.append(BASE_DIR)

def log(m, file=None, timestamped=True, print_also=True):
    curr_time = f"[{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}] "
    log_string = f"{curr_time if timestamped else ''}{m}"
    if file is None:
        print(log_string,file=sys.stderr)
    else:
        print(log_string, file=file)
        if print_also:
            print(log_string, file=sys.stderr)

### Parse Arguments
parser = ArgumentParser(description='DTI_DG Benchmarking.')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--exp-id', required=True, help='Experiment ID', dest='experiment_id')
# parser.add_argument('--model-type',required=True, default="SimplePLMModel", help='Model architecture', dest='model_type')
parser.add_argument('--linear', action='store_true', help='Use a linear model')
parser.add_argument('--coembed', action='store_true', help='Use a coembedding/inner product model')
parser.add_argument('--mol-feat', required=True, help='Molecule featurizer', dest='mol_feat')
parser.add_argument('--prot-feat', required=True, help='Molecule featurizer', dest='prot_feat')
parser.add_argument('--wandb-proj', required=True, help='Weights and Biases Project', dest='wandb_proj')
# parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
#                     help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-4)', dest='lr')
# parser.add_argument('--r', '--replicate', default=0, type=int, help='Replicate', dest='replicate')
parser.add_argument('--d', '--device', default=0, type=int, help='CUDA device', dest='device')

args = parser.parse_args()
if args.linear and args.coembed:
    log('WARNING: --coembed and --linear both set; --linear will be ignored')
device = torch.cuda.set_device(args.device)
outfile = open(f'{LOG_DIR}/{args.experiment_id}_log.txt','w+')
            
def flatten(d):
    d_ = {}
    if not isinstance(d, T.Mapping):
        return d
    for k,v in d.items():
        if isinstance(v, T.Mapping):
            d_flat = flatten(v)
            for k_,v_ in d_flat.items():
                d_[k_] = v_
        else:
            d_[k] = v
    return d_

### Download DTI DG Benchmark Group
from tdc import utils
from tdc.benchmark_group import dti_dg_group

names = utils.retrieve_benchmark_names('DTI_DG_Group')
group = dti_dg_group(path = DATA_DIR)
benchmark = group.get('bindingdb_patent')
name = benchmark['name']
train_val, test = benchmark['train_val'], benchmark['test'] # Natural log transformed (kd/ki/ic50??)

all_drugs = pd.concat([train_val,test]).Drug.values
all_proteins = pd.concat([train_val,test]).Target.values

### Pre-compute drug and protein representations
import src.mol_feats as MOL_FEATS
import src.prot_feats as PROT_FEATS
import src.architectures as ARCHITECTURES
to_disk_path = f"{DATA_DIR}/tdc_bindingdb_patent_train"

# mol_featurizer = Morgan_DC_f()
# prot_featurizer = ProtBert_f()#Prose_f()
mol_featurizer = getattr(MOL_FEATS,args.mol_feat)()
prot_featurizer = getattr(PROT_FEATS,args.prot_feat)()

mol_featurizer.precompute(all_drugs,to_disk_path=to_disk_path,from_disk=True)
prot_featurizer.precompute(all_proteins,to_disk_path=to_disk_path,from_disk=True)

log(f'Using {type(mol_featurizer)} for molecule features',file=outfile)
log(f'Using {type(prot_featurizer)} for protein features',file=outfile)

### Define data loaders
import wandb
import copy
from torch.autograd import Variable
from time import time
from scipy.stats import pearsonr
from plm_dti import DTIDataset, molecule_protein_collate_fn

test_dataset = DTIDataset(
        test.Drug,
        test.Target,
        test.Y,
        mol_featurizer,
        prot_featurizer,
    )

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=lambda x: molecule_protein_collate_fn(x, pad=False))

def get_dataloaders_from_seed(seed, name):
    train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
    
    train_dataset = DTIDataset(
        train.Drug,
        train.Target,
        train.Y,
        mol_featurizer,
        prot_featurizer,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: molecule_protein_collate_fn(x, pad=False))

    valid_dataset = DTIDataset(
        valid.Drug,
        valid.Target,
        valid.Y,
        mol_featurizer,
        prot_featurizer,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: molecule_protein_collate_fn(x, pad=False))
    
    return train_dataloader, valid_dataloader

### Run Training
best_models = {}
timings = {}
pcc_seed = {}

for seed in range(5):
    train_dataloader, valid_dataloader = get_dataloaders_from_seed(seed, name)

    # early stopping
    max_pcc = 0

    if args.coembed:
        model = getattr(ARCHITECTURES, "AffinityCoembedInner")(mol_featurizer._size, prot_featurizer._size).cuda()
    elif args.linear:
        model = getattr(ARCHITECTURES, "AffinityConcatLinear")(mol_featurizer._size, prot_featurizer._size).cuda()
    else:
        model = getattr(ARCHITECTURES, "AffinityEmbedConcat")(mol_featurizer._size, prot_featurizer._size).cuda()
    log(f'Using model type {type(model)}',file=outfile)
    log(str(model),file=outfile)
    torch.backends.cudnn.benchmark = True
    n_epo = args.epochs
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    every_n_val = 1
    loss_history = []
    
    cfg = {
        "data": {'task': 'dti_dg', 'seed': seed},
        "model": str(model),
        "experiment_id": args.experiment_id,
    }
    config = OmegaConf.structured(cfg)
    wandb.init(
            project=args.wandb_proj,
            name=f"{args.experiment_id}_{seed}",
            config=flatten(config),
        )
    wandb.watch(model, log_freq=100)

    tg_len = len(train_dataloader)
    start_time = time()
    for epo in range(n_epo):
        model.train()
        epoch_time_start = time()
        for i, (d, p, label) in enumerate(train_dataloader):

            score = model(d.cuda(), p.cuda())
            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.MSELoss()

            loss = loss_fct(score, label)
            loss_history.append((epo, i, float(loss.cpu().detach().numpy())))
            wandb.log({"train/loss": loss, "epoch": epo,
                           "step": epo*tg_len*args.batch_size + i*args.batch_size
                      })

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i % 1000 == 0):
                log(f'[{seed}] Training at Epoch {epo+1} iteration {i} with loss {loss.cpu().detach().numpy()}',
                    file=outfile)

        epoch_time_end = time()
        if epo % every_n_val == 0:
            with torch.set_grad_enabled(False):
                pred_list = []
                lab_list = []
                model.eval()
                for i, (d, p, label) in enumerate(valid_dataloader):
                    score = model(d.cuda(), p.cuda())
                    score = score.detach().cpu().numpy()
                    label = label.detach().cpu().numpy()
                    pred_list.extend(score)
                    lab_list.extend(label)

                pred_list = torch.tensor(pred_list)
                lab_list = torch.tensor(lab_list)
                val_pcc = pearsonr(pred_list, lab_list)[0]
                wandb.log({"epoch": epo,
                           "val/pcc": float(val_pcc),
                           "Charts/epoch_time": (epoch_time_end - epoch_time_start)/every_n_val
                  })
                if val_pcc > max_pcc:
                    model_max = copy.deepcopy(model)
                    max_pcc = val_pcc
                log(f'[{seed}] Validation at Epoch {epo+1}: PCC={val_pcc}',file=outfile)
        end_time = time()
        
    best_models[seed] = (model_max, max_pcc)
    timings[seed] = end_time - start_time
    torch.save(model_max, f"{MODEL_BASE_DIR}/{args.experiment_id}_best_model.sav")
    
    pred_list = []

    best_mod_ev = best_models[seed][0]
    best_mod_ev.eval()
    with torch.no_grad():
        for i, (d, p, label) in enumerate(test_dataloader):
            score = best_mod_ev(d.cuda(), p.cuda())
            score = score.detach().cpu().numpy()
            pred_list.extend(score)

    pred_list = np.array(pred_list)
    predictions = {name: pred_list}
    
    out = group.evaluate(predictions)
    pcc_seed[seed] = out
    log(f'{seed}: PCC={out[name]["pcc"]}',file=outfile)
    wandb.log({"test/pcc": out[name]["pcc"]})
    wandb.finish()

log(f'Average PCC: {sum([pcc_seed[s][name]["pcc"] for s in range(5)])/5}',file=outfile)

### Print Model Info
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
nparams = sum([np.prod(p.size()) for p in model_parameters])
log(f'Model #Parameters: {nparams}',file=outfile)
log(f'Avg. Train Time ({n_epo} epochs): {sum(timings.values())/5} seconds',file=outfile)

outfile.close()