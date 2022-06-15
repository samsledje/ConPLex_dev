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
# from dpatch import PB_Embed
from torch.nn.utils.rnn import pad_sequence

from . import architectures as ARCHITECTURES
from . import prot_feats as PROT_FEATURIZERS
from . import mol_feats as MOL_FEATURIZERS

# MOL_FEATURIZERS = {
#     "Morgan_f": Morgan_f,
#     "Mol2Vec_f": Mol2Vec_f,
# }

# PROT_FEATURIZERS = {
#     "Random_f": Random_f,
#     "BeplerBerger_f": BeplerBerger_f,
#     "BeplerBerger_DSCRIPT_f": BeplerBerger_DSCRIPT_f,
#     "BeplerBerger_DSCRIPT_cat_f": BeplerBerger_DSCRIPT_cat_f,
#     "Prose_f": Prose_f,
#     "ESM_f": ESM_f,
#     "ESM_DSCRIPT_f": ESM_DSCRIPT_f,
#     "ESM_DSCRIPT_cat_f": ESM_DSCRIPT_cat_f,
#     "ProtBert_f": ProtBert_f,
#     "ProtBert_DSCRIPT_f": ProtBert_DSCRIPT_f,
#     "ProtBert_DSCRIPT_cat_f": ProtBert_DSCRIPT_cat_f,
#     # "DPatch_f": DPatch_f,
# }

#####################################
# Protein Interface Representations #
#####################################

# class DPatch_f:
#     def __init__(self, pool=True):
#         from dscript.language_model import lm_embed
#         from dscript.pretrained import get_pretrained
#         self.use_cuda = True
#         self.pool = pool
#         self._size = 50
#         self._max_len = 800
#         self.precomputed = False

#         self._embed = lm_embed
#         self._dscript_model = get_pretrained("human_v1")
#         self._dscript_model.use_cuda = self.use_cuda
#         self._pbe = PB_Embed.load_from_checkpoint("/afs/csail.mit.edu/u/s/samsl/Work/Interface_Prediction/src/pb_embed_20210425.ckpt")
#         if self.use_cuda:
#             self._dscript_model = self._dscript_model.cuda()
#             self._pbe = self._pbe.cuda()

#     def precompute(self, seqs, to_disk_path=True, from_disk=True):
#         print("--- precomputing dpatch protein featurizer ---")
#         assert not self.precomputed
#         precompute_path = f"{to_disk_path}_DPATCH_PROTEINS{'_STACKED' if not self.pool else ''}.pk"
#         if from_disk and os.path.exists(precompute_path):
#             print("--- loading from disk ---")
#             self.prot_embs = pk.load(open(precompute_path,"rb"))
#         else:
#             self.prot_embs = {}
#             for sq in tqdm(seqs):
#                 if sq in self.prot_embs:
#                     continue
#                 self.prot_embs[sq] = self._transform(sq)

#             if to_disk_path is not None and not os.path.exists(precompute_path):
#                 print(f'--- saving protein embeddings to {precompute_path} ---')
#                 pk.dump(self.prot_embs, open(precompute_path,"wb+"))
#         self.precomputed = True

#     @lru_cache(maxsize=5000)
#     def _transform(self, seq):
#         if len(seq) > self._max_len:
#             seq = seq[:self._max_len]

#         with torch.no_grad():
#             lm_emb = self._embed(seq, use_cuda=self.use_cuda)
#             if self.use_cuda:
#                 lm_emb = lm_emb.cuda()
#             ds_emb = self._dscript_model.embedding(lm_emb)
#             ds_emb = ds_emb.squeeze().mean(axis=0)
#             return self._pbe(ds_emb)

#     def __call__(self, seq):
#         if self.precomputed:
#             return self.prot_embs[seq]
#         else:
#             return self._transform(seq)

##################
# Data Set Utils #
##################

def molecule_protein_collate_fn(args, pad=False):
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

class DTIDataset(Dataset):
    def __init__(self,smiles, sequences, labels, mfeats, pfeats):
        assert len(smiles) == len(sequences)
        assert len(sequences) == len(labels)
        self.smiles = smiles
        self.sequences = sequences
        self.labels = labels

        self.mfeats = mfeats
        self.pfeats = pfeats

    def __len__(self):
        return len(self.smiles)

    @property
    def shape(self):
        return self.mfeats._size, self.pfeats._size

    def __getitem__(self, i):
        memb = self.mfeats(self.smiles[i])
        pemb = self.pfeats(self.sequences[i])
        lab = torch.tensor(self.labels[i])

        return memb, pemb, lab
    
class ContrastiveDataset(Dataset):
    def __init__(self, contrastive_df, mfeats, pfeats):
        self.mfeats = mfeats
        self.pfeats = pfeats
        self.contrastive_df = contrastive_df
        
        self.anchors = contrastive_df.Anchor
        self.positives = contrastive_df.Positive
        self.negatives = contrastive_df.Negative

    def __len__(self):
        return len(self.contrastive_df)

    @property
    def shape(self):
        return self.mfeats._size, self.pfeats._size

    def __getitem__(self, i):
        
        anchorEmb = self.pfeats(self.anchors[i])
        positiveEmb = self.mfeats(self.positives[i])
        negativeEmb = self.mfeats(self.negatives[i])

        return anchorEmb, positiveEmb, negativeEmb

#################
# API Functions #
#################

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
        mol_feats = getattr(MOL_FEATURIZERS, mol_feat)()
    except AttributeError:
        raise ValueError(f"Specified molecule featurizer {mol_feat} is not supported")
    try:
        prot_feats = getattr(PROT_FEATURIZERS, prot_feat)(pool=pool)
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
    


def get_dataloaders_contrastive(train_df,
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
        mol_feats = getattr(MOL_FEATURIZERS, mol_feat)()
    except AttributeError:
        raise ValueError(f"Specified molecule featurizer {mol_feat} is not supported")
    try:
        prot_feats = getattr(PROT_FEATURIZERS, prot_feat)(pool=pool)
    except AttributeError:
        raise ValueError(f"Specified protein featurizer {prot_feat} is not supported")
    if precompute:
        mol_feats.precompute(all_smiles,to_disk_path=to_disk_path,from_disk=True)
        prot_feats.precompute(all_sequences,to_disk_path=to_disk_path,from_disk=True)

    loaders = {}
    contrastive_df = make_contrastive(train_df)
    contrastive_dset = ContrastiveDataset(contrastive_df, mol_feats, prot_feats)
    contrastive_dataloader = DataLoader(contrastive_dset,
                                batch_size=batch_size,shuffle=shuffle,
                                num_workers=num_workers,collate_fn=lambda x: molecule_protein_collate_fn(x, pad=not pool)
                               )
    loaders["contrastive"] = contrastive_dataloader
    
    for set_name in ["train", "val", "test"]:
        smiles, sequences, labels = df_values[set_name]

        dataset = DTIDataset(smiles, sequences, labels, mol_feats, prot_feats)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,shuffle=shuffle,
                                num_workers=num_workers,collate_fn=lambda x: molecule_protein_collate_fn(x, pad=not pool)
                               )
        loaders[set_name] = dataloader

    return (loaders["train"], loaders["contrastive"], loaders["val"], loaders["test"], mol_feats._size, prot_feats._size)

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
        mol_feats = getattr(MOL_FEATURIZERS, mol_feat)()
    except AttributeError:
        raise ValueError(f"Specified molecule featurizer {mol_feat} is not supported")
    try:
        prot_feats = getattr(PROT_FEATURIZERS, prot_feat)(pool=pool)
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
        return getattr(ARCHITECTURES, model_type)(**model_kwargs)
    except AttributeError:
        raise ValueError("Specified model is not supported")
        


