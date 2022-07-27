DEBUG = True

from src.architectures import SimpleCoembedding, GoldmanCPI
from src.featurizers import ProtBertFeaturizer, MorganFeaturizer
from src.utils import config_logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
import os
import numpy as np
import pandas as pd
import pickle as pk
import typing as T
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from pprint import PrettyPrinter
from collections.abc import Iterable

from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score

from dataclasses import dataclass
from src.data import BinaryDataset, drug_target_collate_fn

TASK_PATH = {
    "halogenase": "./dataset/EnzPred/halogenase_NaBr_binary.csv",
    "gt": "./dataset/EnzPred/gt_acceptors_achiral_binary.csv",
    "bkace": "./dataset/EnzPred/duf_binary.csv",
    "esterase": "./dataset/EnzPred/esterase_binary.csv",
    "phosphatase": "./dataset/EnzPred/phosphatase_chiral_binary.csv",
    "kinase": "./dataset/EnzPred/davis_filtered.csv",
}

N_SPLITS = {
    "halogenase": 10,
    "gt": 10,
    "bkace": 10,
    "esterase": 10,
    "phosphatase": 10,
    "kinase": 10,
}


@dataclass
class Config:
    drug_shape: int = 2048
    target_shape: int = 1024
    latent_shape: int = 1024
    lr: float = 1e-4
    epochs: int = 25
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    model_class: nn.Module = SimpleCoembedding
    checkpoint: T.Union[Path, None] = Path(
        "/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/best_models/ProdModels/prod_within_state.pt"
    )
    # checkpoint: T.Union[Path, None] = None,


conf = Config()


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def create_model(conf):
    model = conf.model_class(
        conf.drug_shape,
        conf.target_shape,
        latent_dimension=conf.latent_shape,
        latent_distance="Cosine",
        classify=True,
    ).to(conf.device)

    if isinstance(conf.checkpoint, Path):
        try:
            state_dict = torch.load(conf.checkpoint)
        except FileNotFoundError as e:
            logg.debug(e)
            logg.error(f"File {conf.checkpoint} not found")
            sys.exit(1)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logg.debug(e)
            logg.error(
                f"Model not compatible with checkpoint {conf.checkpoint}"
            )
            sys.exit(1)

    return model


def create_data(dataframe, conf, drug_feat, target_feat):

    bdataset = BinaryDataset(
        dataframe[conf.drug_col],
        dataframe[conf.target_col],
        dataframe[conf.label_col],
        drug_feat,
        target_feat,
    )
    bdataloader = DataLoader(
        bdataset,
        batch_size=conf.batch_size,
        shuffle=conf.shuffle,
        num_workers=conf.num_workers,
        collate_fn=drug_target_collate_fn,
    )
    return bdataloader


def step(model, batch, device=None):

    if device is None:
        device = torch.device("cpu")

    drug, target, label = batch

    pred = model(drug.to(device), target.to(device))
    label = Variable(torch.from_numpy(np.array(label)).float()).to(device)
    return pred, label


def train_model(model, dataloader, conf):

    opt = torch.optim.Adam(model.parameters(), lr=conf.lr)
    loss_fct = torch.nn.BCELoss()
    model.train()

    for epo in tqdm(range(conf.epochs), leave=False, desc="Epoch"):

        for i, batch in tqdm(
            enumerate(dataloader),
            leave=False,
            desc="Batch",
            total=len(dataloader),
        ):
            # for i, batch in enumerate(dataloader):

            pred, label = step(model, batch, conf.device)
            pred = pred.view(-1)
            label = label.view(-1)
            try:
                loss = loss_fct(pred, label)
            except ValueError as e:
                logg.error(e)
                logg.debug(pred)
                logg.debug(label)
                logg.debug(pred.shape)
                logg.debug(label.shape)
                sys.exit(1)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def eval_model(model, dataloader, conf):
    model.eval()

    preds = []
    labels = []

    with torch.set_grad_enabled(False):
        for i, batch in enumerate(dataloader):

            pred, label = step(model, batch, conf.device)
            preds.append(pred.view(-1))
            labels.append(label.view(-1))

    preds = torch.cat(preds).detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()

    return preds, labels


usage_help = "usage: python test_EnzPred.py [task] [logfile]"
try:
    conf.enzyme_type = sys.argv[1]
    conf.logfile = Path(sys.argv[2])
    conf.savedir = conf.logfile.parent / Path(conf.enzyme_type)
except ValueError:
    raise ValueError(usage_help)
    sys.exit(0)
assert conf.enzyme_type in N_SPLITS.keys(), usage_help
conf.device = torch.device("cuda:0")

if DEBUG:
    log_level = 3
else:
    log_level = 2

logg = config_logger(
    conf.logfile.with_suffix(".log"),
    "%(asctime)s [%(levelname)s] %(message)s",
    log_level,
    use_stdout=False,
)

data_file = TASK_PATH[conf.enzyme_type]
full_df = pd.read_csv(data_file, index_col=0)
conf.target_col = full_df.columns[0]
conf.drug_col = full_df.columns[1]
conf.label_col = full_df.columns[2]

substrates = full_df[conf.drug_col].unique()
enzymes = full_df[conf.target_col].unique()

conf.n_splits = (
    len(enzymes)
    if N_SPLITS[conf.enzyme_type] == "N"
    else N_SPLITS[conf.enzyme_type]
)
kfsplitter = KFold(conf.n_splits)

logg.info("Loading data")
os.makedirs(conf.savedir, exist_ok=True)

drug_feat = MorganFeaturizer(save_dir=conf.savedir).to(conf.device)
drug_feat.preload(substrates)

target_feat = ProtBertFeaturizer(save_dir=conf.savedir).to(conf.device)
target_feat.preload(enzymes)

pp = PrettyPrinter()
conf_pprint = pp.pformat(vars(conf))
logg.debug(vars(conf)["model_class"])
logg.info(conf_pprint)

all_predictions = defaultdict(list)
all_labels = defaultdict(list)

model = create_model(conf)
logg.info(model)

for i, (train_ind, test_ind) in tqdm(
    enumerate(kfsplitter.split(enzymes)), desc="Split", total=conf.n_splits
):

    train_enzymes = [enzymes[i] for i in train_ind]
    train_df = full_df[full_df[conf.target_col].isin(train_enzymes)]
    train_dataloader = create_data(train_df, conf, drug_feat, target_feat)

    test_enzymes = [enzymes[i] for i in test_ind]
    test_df = full_df[full_df[conf.target_col].isin(test_enzymes)]
    # test_dataloader = create_data(test_df, conf, drug_feat, target_feat)

    model = create_model(conf)
    model = train_model(model, train_dataloader, conf)

    for curr_task in tqdm(substrates, leave=False, desc="Task"):
        task_df = test_df[test_df[conf.drug_col] == curr_task]
        if len(task_df):
            task_dataloader = create_data(
                task_df, conf, drug_feat, target_feat
            )

            prd, lab = eval_model(model, task_dataloader, conf)
            all_labels[curr_task].append(lab)
            all_predictions[curr_task].append(prd)
        else:
            continue

task_aupr_dict = {}
skipped = 0

for curr_task in tqdm(substrates, leave=False, desc="Task Evaluation"):
    try:
        lab = np.concatenate(all_labels[curr_task])
        prd = np.concatenate(all_predictions[curr_task])

        pct_lab = lab.sum() / len(lab)
        if (0.1 > pct_lab) or (0.9 < pct_lab):
            logg.debug(f"Skipping {curr_task} ({pct_lab} pct. positive)")
            skipped += 1
            continue

        curr_aupr = average_precision_score(lab, prd)
        task_aupr_dict[curr_task] = curr_aupr
        logg.info(f"Substrate {curr_task} AUPR: {curr_aupr}")
    except ValueError as e:
        logg.error(e)
        logg.debug(curr_task)
        logg.debug(lab)
        logg.debug(prd)

task_auprs = list(task_aupr_dict.values())
avg_of_avg = np.nanmean(task_auprs)
logg.info(f"Skipped {skipped} tasks for data imbalance")
logg.info(f"Kept {len(substrates) - skipped} tasks")
logg.info(f"Average per-task AUPR: {avg_of_avg}")

try:
    labels = list(flatten(list(all_labels.values())))
    preds = list(flatten(list(all_predictions.values())))
    overall_aupr = average_precision_score(labels, preds)
    logg.info(f"Overall AUPR: {overall_aupr}")
except ValueError as e:
    logg.error(e)
    # logg.debug(all_labels.values())
    # logg.debug(all_predictions.values())

with open(conf.logfile.with_suffix(".results.pk"), "wb") as fi:
    pk.dump(task_auprs, fi)
