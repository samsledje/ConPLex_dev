import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dscript
import os
import pickle as pk
import pandas as pd
import pytorch_lightning as pl

from types import SimpleNamespace
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
from numpy.random import choice
from torch.nn.utils.rnn import pad_sequence

from . import architectures as dti_architecture
from . import protein as protein_features
from . import molecule as molecule_features

from .featurizers import Featurizer
from pathlib import Path
import typing as T

def get_task_dir(task_name):
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'
    elif task_name.lower() == 'biosnap_prot':
        return './dataset/BIOSNAP/unseen_protein'
    elif task_name.lower() == 'biosnap_mol':
        return './dataset/BIOSNAP/unseen_drug'

class BinaryDataset(Dataset):
    def __init__(self,
                 drugs,
                 targets,
                 labels,
                 drug_featurizer: Featurizer,
                 target_featurizer: Featurizer
                ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i):
        drug = self.drug_featurizer(self.drugs[i])
        target = self.target_featurizer(self.targets[i])
        label = torch.tensor(self.labels[i])

        return drug, target, label
    
class ContrastiveDataset(Dataset):
    def __init__(self,
                 anchors,
                 positives,
                 negatives,
                 drug_featurizer: Featurizer,
                 target_featurizer: Featurizer
                ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        
        anchorEmb = self.target_featurizer(self.anchors[i])
        positiveEmb = self.drug_featurizer(self.positives[i])
        negativeEmb = self.drug_featurizer(self.negatives[i])

        return anchorEmb, positiveEmb, negativeEmb

class DTIDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 drug_featurizer: Featurizer,
                 target_featurizer: Featurizer,
                 device: torch.device = torch.device("cpu"),
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 header = 0,
                 index_col = 0,
                 sep = ","
                ):
        
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn
        }
        
        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep
        }
        
        self._device = device
        
        self._data_dir = Path(data_dir)
        self._train_path = Path("train.csv")
        self._val_path = Path("val.csv")
        self._test_path = Path("test.csv")
        
        self._drug_column = "SMILES"
        self._target_column = "Target Sequence"
        self._label_column = "Label"
        
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer
        
    def setup(self, stage: T.Optional[str] = None):
        
        self.df_train = pd.read_csv(self._data_dir / self._train_path,
                                    **self._csv_kwargs
                                   )
        self.df_val = pd.read_csv(self._data_dir / self._val_path,
                                  **self._csv_kwargs
                                 )
        self.df_test = pd.read_csv(self._data_dir / self._test_path,
                                   **self._csv_kwargs
                                  )
        self._dataframes = [self.df_train, self.df_val, self.df_test]
        
        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()
        
        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)
            
        self.drug_featurizer.precompute(all_drugs)
        self.drug_featurizer.cpu()
        
        self.target_featurizer.precompute(all_targets)
        self.target_featurizer.cpu()
        
        if stage == "fit" or stage is None:    
            self.data_train = BinaryDataset(self.df_train[self._drug_column],
                                            self.df_train[self._target_column],
                                            self.df_train[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
            
            self.data_val = BinaryDataset(self.df_val[self._drug_column],
                                            self.df_val[self._target_column],
                                            self.df_val[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
                
        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(self.df_test[self._drug_column],
                                            self.df_test[self._target_column],
                                            self.df_test[self._label_column],
                                            self.drug_featurizer,
                                            self.target_featurizer
                                           )
            
    def train_dataloader(self):
        return DataLoader(self.data_train, 
                          **self._loader_kwargs
                         )

    def val_dataloader(self):
        return DataLoader(self.data_val,
                        **self._loader_kwargs
                         )

    def test_dataloader(self):
        return DataLoader(self.data_test,
                         **self._loader_kwargs
                         )
            

def drug_target_collate_fn(args, pad=False):
    """
    Collate function for PyTorch data loader.

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    """
    memb = [a[0] for a in args]
    pemb = [a[1] for a in args]
    labs = [a[2] for a in args]

    if pad:
        proteins = pad_sequence(pemb,batch_first=True)
    else:
        proteins = torch.stack(pemb, 0)
    molecules = torch.stack(memb, 0)
    affinities = torch.stack(labs, 0)

    return molecules, proteins, affinities

def make_contrastive(df,
                     mol_col = 'SMILES',
                     prot_col = 'Target Sequence',
                     label_col = 'Label',
                     n_neg_per = 5
                    ): 
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]
    
    contrastive = []

    for _,r in pos_df.iterrows():
        for _ in range(n_neg_per):
            contrastive.append((r[prot_col], r[mol_col], choice(neg_df[mol_col])))

    contrastive = pd.DataFrame(contrastive,columns=['Anchor','Positive','Negative'])
    return contrastive

def get_dataloaders_dude(train_set,
                         batch_size,
                         shuffle,
                         num_workers,
                         mol_feat,
                         prot_feat,
                         pool = True,
                         precompute = True,
                         to_disk_path=None,
                         device=0,
                         n_neg_per = 50
                        ):
    
    full_dude = pd.read_csv('./dataset/DUDe/full.tsv',sep='\t')
    train_dude = full_dude[full_dude.Target_ID.isin(train_set)]
    contrastive_dude = make_contrastive(train_dude,
                                        mol_col = 'Molecule_SMILES',
                                        prot_col = 'Target_Seq',
                                        label_col = 'Label',
                                        n_neg_per = n_neg_per)
    
    all_smiles = list(train_dude.Molecule_SMILES.unique())
    all_sequences = list(train_dude.Target_Seq.unique())
    try:
        mol_feats = getattr(molecule_features, mol_feat)()
    except AttributeError:
        raise ValueError(f"Specified molecule featurizer {mol_feat} is not supported")
    try:
        prot_feats = getattr(protein_features, prot_feat)(pool=pool)
    except AttributeError:
        raise ValueError(f"Specified protein featurizer {prot_feat} is not supported")
    if precompute:
        mol_feats.precompute(all_smiles,to_disk_path=to_disk_path,from_disk=True)
        prot_feats.precompute(all_sequences,to_disk_path=to_disk_path,from_disk=True)
    
    contrastive_dset = ContrastiveDataset(contrastive_dude, mol_feats, prot_feats)
    contrastive_dataloader = DataLoader(contrastive_dset,
                                batch_size=batch_size,shuffle=shuffle,
                                num_workers=num_workers,collate_fn=lambda x: molecule_protein_collate_fn(x, pad=not pool)
                               )
    return contrastive_dataloader

def get_dataloaders(train_df,
                    val_df,
                    test_df,
                    batch_size,
                    shuffle,
                    num_workers,
                    mol_feat,
                    prot_feat,
                    pool = True,
                    precompute=True,
                    to_disk_path=None,
                    device=0,
                  ):

    df_values = {}
    all_smiles = []
    all_sequences = []
    for df, set_name in zip([train_df, val_df, test_df], ["train", "val", "test"]):
        all_smiles.extend(df["SMILES"])
        all_sequences.extend(df["Target Sequence"])
        df_thin = df[["SMILES","Target Sequence","Label"]]
        df_values[set_name] = (df["SMILES"], df["Target Sequence"], df["Label"])

    try:
        mol_feats = getattr(molecule_features, mol_feat)()
    except AttributeError:
        raise ValueError(f"Specified molecule featurizer {mol_feat} is not supported")
    try:
        prot_feats = getattr(protein_features, prot_feat)(pool=pool)
    except AttributeError:
        raise ValueError(f"Specified protein featurizer {prot_feat} is not supported")
    if precompute:
        mol_feats.precompute(all_smiles,to_disk_path=to_disk_path,from_disk=True)
        prot_feats.precompute(all_sequences,to_disk_path=to_disk_path,from_disk=True)

    loaders = {}
    for set_name in ["train", "val", "test"]:
        smiles, sequences, labels = df_values[set_name]

        dataset = DTIDataset(smiles, sequences, labels, mol_feats, prot_feats)
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=lambda x: molecule_protein_collate_fn(x, pad=not pool))
        loaders[set_name] = dataloader

    return tuple([loaders["train"], loaders["val"], loaders["test"], mol_feats._size, prot_feats._size])