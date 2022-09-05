DEBUG = True

from src.architectures import SimpleCoembedding, GoldmanCPI
from src.featurizers import (
    Featurizer,
    ProtBertFeaturizer,
    MorganFeaturizer,
    ESMFeaturizer,
)
from src.utils import (
    MarginScheduledLossFunction,
    set_random_seed,
    config_logger,
)
from src.data import (
    BinaryDataset,
    ContrastiveDataset,
    drug_target_collate_fn,
    make_contrastive,
)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
import os
import wandb
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

from sklearn.linear_model import Ridge

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
    contrastive: bool = True
    drug_shape: int = 2048
    # target_shape: int = 1280
    target_shape: int = 1024
    latent_shape: int = 1024
    lr: float = 1e-4
    epochs: int = 25
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    model_class: str = "SingleTask"
    target_featurizer: Featurizer = ESMFeaturizer
    n_neg_per: int = 25
    replicate: int = 3


conf = Config()


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def create_model_ridge(conf):
    model = Ridge()
    return model


if __name__ == "__main__":
    usage_help = "usage: python test_EnzPred.py [task] [logfile] [prefix]"
    try:
        conf.enzyme_type = sys.argv[1]
        conf.task = conf.enzyme_type
        conf.logfile = Path(sys.argv[2])
        conf.savedir = conf.logfile.parent.parent / Path(conf.enzyme_type)
        # conf.prefix = sys.argv[3]
    except ValueError:
        raise ValueError(usage_help)
        sys.exit(0)
    assert conf.enzyme_type in N_SPLITS.keys(), usage_help
    conf.device = torch.device("cuda:0")

    set_random_seed(conf.replicate)

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
    kfsplitter = KFold(
        conf.n_splits, shuffle=True, random_state=conf.replicate
    )

    logg.info("Loading data")
    os.makedirs(conf.savedir, exist_ok=True)

    target_feat = conf.target_featurizer(save_dir=conf.savedir).to(conf.device)
    target_feat = ESMFeaturizer(save_dir=conf.savedir).to(conf.device)
    target_feat.preload(enzymes)

    pp = PrettyPrinter()
    conf_pprint = pp.pformat(vars(conf))
    logg.info(conf_pprint)

    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)

    wandb.init(
        project="EnzPred",
        name=conf.logfile.stem,
        config=vars(conf),
    )

    skipped = 0
    task_aupr_dict = {}
    for curr_task in tqdm(substrates, desc="Task Evaluation"):

        task_df = full_df[full_df[conf.drug_col] == curr_task]
        lab = task_df[conf.label_col].values
        pct_lab = lab.sum() / len(lab)
        if (0.1 > pct_lab) or (0.9 < pct_lab):
            logg.debug(f"Skipping {curr_task} ({pct_lab} pct. positive)")
            skipped += 1
            continue

        for i, (train_ind, test_ind) in tqdm(
            enumerate(kfsplitter.split(enzymes)),
            desc="Split",
            total=conf.n_splits,
            leave=False,
        ):

            train_enzymes = [enzymes[i] for i in train_ind]
            test_enzymes = [enzymes[i] for i in test_ind]

            train_df = task_df[task_df[conf.target_col].isin(train_enzymes)]
            test_df = task_df[task_df[conf.target_col].isin(test_enzymes)]

            model = create_model_ridge(conf)

            train_X = []
            for i, r in train_df.iterrows():
                train_X.append(target_feat(r[conf.target_col]))
            train_X = torch.stack(train_X, 0).detach().cpu().numpy()
            train_Y = train_df[conf.label_col].values
            assert len(train_X) == len(train_Y)

            model.fit(train_X, train_Y)

            test_X = []
            for i, r in test_df.iterrows():
                test_X.append(target_feat(r[conf.target_col]))
            test_X = torch.stack(test_X).detach().cpu().numpy()
            test_Y = test_df[conf.label_col].values

            prd = model.predict(test_X)
            all_labels[curr_task].append(test_Y)
            all_predictions[curr_task].append(prd)

        labels = list(flatten(all_labels[curr_task]))
        predictions = list(flatten(all_predictions[curr_task]))
        curr_aupr = average_precision_score(labels, predictions)
        logg.info(f"Substrate {curr_task} AUPR: {curr_aupr}")
        task_aupr_dict[curr_task] = curr_aupr

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

    wandb.log({"test/avg_aupr": avg_of_avg, "test/overall_aupr": overall_aupr})

    with open(conf.logfile.with_suffix(".results.pk"), "wb") as fi:
        pk.dump(task_auprs, fi)
