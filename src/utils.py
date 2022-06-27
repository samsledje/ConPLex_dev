import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dscript
import os
import pickle as pk
import pandas as pd
from types import SimpleNamespace
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
from numpy.random import choice
from torch.nn.utils.rnn import pad_sequence

from . import architecture as dti_architecture
from . import protein as protein_features
from . import molecule as molecule_features

def get_config(experiment_id, mol_feat, prot_feat):
    data_cfg = {
        "batch_size":32,
        "num_workers":0,
        "precompute":True,
        "mol_feat": mol_feat,
        "prot_feat": prot_feat,
    }
    model_cfg = {
        # "latent_size": 1024,
        # "distance_metric": "Cosine"
    }
    training_cfg = {
        "n_epochs": 50,
        "every_n_val": 1,
    }
    cfg = {
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "experiment_id": experiment_id
    }

    return OmegaConf.structured(cfg)

def get_model(model_type, **model_kwargs):
    try:
        return getattr(dti_architecture, model_type)(**model_kwargs)
    except AttributeError:
        raise ValueError("Specified model is not supported")