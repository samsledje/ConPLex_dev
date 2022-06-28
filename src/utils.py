import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

import logging as lg

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from . import architectures as dti_architecture
from . import protein as protein_features
from . import molecule as molecule_features

logLevels = {0: lg.ERROR, 1: lg.WARNING, 2: lg.INFO, 3: lg.DEBUG}

def config_logger(file, fmt, level=2, use_stdout=True):
    module_logger = lg.getLogger("DTI")
    module_logger.setLevel(logLevels[level])
    formatter = lg.Formatter(fmt)
    
    if file is not None:
        fh = lg.FileHandler(file)
        fh.setFormatter(formatter)
        module_logger.addHandler(fh)

    if use_stdout:
        sh = lg.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        module_logger.addHandler(sh)

    return module_logger

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None

def smiles2morgan(s, radius = 2, nBits = 2048):
    """Convert smiles into Morgan Fingerprint. 
    Args: 
      smiles: str
      radius: int (default: 2)
      nBits: int (default: 1024)
    Returns:
      fp: numpy.array
    """  
    try:
        s = canonicalize(s)
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except Exception as e:
        print(e)
        print(f'rdkit not found this smiles for morgan: {s} convert to all 0 features')
        features = np.zeros((nBits, ))
    return features

def get_config(experiment_id, mol_feat, prot_feat):
    data_cfg = {
        "batch_size": 32,
        "num_workers": 0,
        "precompute":True,
        "mol_feat": mol_feat,
        "prot_feat": prot_feat,
    }
    model_cfg = {
        "latent_size": 1024,
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