import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dscript
import os
import numpy as np
import pickle as pk
from types import SimpleNamespace
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
# from dpatch import PB_Embed
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

PRECOMPUTED_MOLECULE_PATH = "precomputed_molecules.pk"

#################################
# Sanity Check Null Featurizers #
#################################

class Random_f:
    def __init__(self, size = 1024, pool=True):
        self.use_cuda = True
        self._size = size

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        pass

    def _transform(self, seq):
        return torch.rand(self._size).cuda()

    def __call__(self, seq):
        return self._transform(seq)
    
class Null_f:
    def __init__(self, size = 1024, pool=True):
        self.use_cuda = True
        self._size = size

    def precompute(self, seqs, to_disk_path=True, from_disk=True):
        pass

    def _transform(self, seq):
        return torch.zeros(self._size).cuda()

    def __call__(self, seq):
        return self._transform(seq)

#########################
# Molecular Featurizers #
#########################

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
    except:
        print(f'rdkit not found this smiles for morgan: {s} convert to all 0 features')
        features = np.zeros((nBits, ))
    return features

class Morgan_f:
    def __init__(self,
                 size=2048,
                 radius=2,
                ):
        import deepchem as dc
        self._morgan_featurizer = lambda x: smiles2morgan(x, radius=radius, nBits=size)
        self._size = size
        self.use_cuda = True
        self.precomputed = False

    def precompute(self, smiles, to_disk_path=None, from_disk=True):
        print("--- precomputing morgan molecule featurizer ---")
        assert not self.precomputed

        if from_disk and os.path.exists(f"{to_disk_path}_Morgan_MOLECULES.pk"):
            print("--- loading from disk ---")
            self.mol_embs = pk.load(open(f"{to_disk_path}_Morgan_MOLECULES.pk","rb"))
        else:
            self.mol_embs = {}
            for sm in tqdm(smiles):
                if sm in self.mol_embs:
                    continue
                m_emb = self._transform(sm)
                if len(m_emb) != self._size:
                    m_emb = torch.zeros(self._size)
                    if self.use_cuda:
                        m_emb = m_emb.cuda()
                self.mol_embs[sm] = m_emb
            if to_disk_path is not None and not os.path.exists(f"{to_disk_path}_Morgan_MOLECULES.pk"):
                print(f'--- saving morgans to {f"{to_disk_path}_Morgan_MOLECULES.pk"} ---')
                pk.dump(self.mol_embs, open(f"{to_disk_path}_Morgan_MOLECULES.pk","wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, smile):
        tens = torch.from_numpy(self._morgan_featurizer(smile)).squeeze().float()
        if self.use_cuda:
            tens = tens.cuda()

        return tens

    def __call__(self, smile):
        if self.precomputed:
            return self.mol_embs[smile]
        else:
            return self._transform(smile)
        
class Morgan_DC_f:
    def __init__(self,
                 size=2048,
                 radius=2,
                ):
        import deepchem as dc
        self._dc_featurizer = dc.feat.CircularFingerprint()
        self._size = size
        self.use_cuda = True
        self.precomputed = False

    def precompute(self, smiles, to_disk_path=None, from_disk=True):
        print("--- precomputing morgan_DC molecule featurizer ---")
        assert not self.precomputed

        if from_disk and os.path.exists(f"{to_disk_path}_Morgan_DC_MOLECULES.pk"):
            print("--- loading from disk ---")
            self.mol_embs = pk.load(open(f"{to_disk_path}_Morgan_DC_MOLECULES.pk","rb"))
        else:
            self.mol_embs = {}
            for sm in tqdm(smiles):
                if sm in self.mol_embs:
                    continue
                m_emb = self._transform(sm)
                if len(m_emb) != self._size:
                    m_emb = torch.zeros(self._size)
                    if self.use_cuda:
                        m_emb = m_emb.cuda()
                self.mol_embs[sm] = m_emb
            if to_disk_path is not None and not os.path.exists(f"{to_disk_path}_Morgan_DC_MOLECULES.pk"):
                print(f'--- saving morgans to {f"{to_disk_path}_Morgan_DC_MOLECULES.pk"} ---')
                pk.dump(self.mol_embs, open(f"{to_disk_path}_Morgan_DC_MOLECULES.pk","wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, smile):
        tens = torch.from_numpy(self._dc_featurizer.featurize([smile])).squeeze().float()
        if self.use_cuda:
            tens = tens.cuda()

        return tens

    def __call__(self, smile):
        if self.precomputed:
            return self.mol_embs[smile]
        else:
            return self._transform(smile)

class Mol2Vec_f:
    def __init__(self,
                 radius=1,
                ):
        import deepchem as dc
        self._dc_featurizer = dc.feat.Mol2VecFingerprint()
        self._size = 300
        self.use_cuda = True
        self.precomputed = False

    def precompute(self, smiles, to_disk_path=None, from_disk=True):
        print("--- precomputing mol2vec molecule featurizer ---")
        assert not self.precomputed

        if from_disk and os.path.exists(f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk"):
            print("--- loading from disk ---")
            self.mol_embs = pk.load(open(f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk","rb"))
        else:
            self.mol_embs = {}
            for sm in tqdm(smiles):
                if sm in self.mol_embs:
                    continue
                m_emb = self._transform(sm)
                if len(m_emb) != self._size:
                    m_emb = torch.zeros(self._size)
                    if self.use_cuda:
                        m_emb = m_emb.cuda()
                self.mol_embs[sm] = m_emb
            if to_disk_path is not None and not os.path.exists(f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk"):
                print(f'--- saving morgans to {f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk"} ---')
                pk.dump(self.mol_embs, open(f"{to_disk_path}_Mol2Vec_PRECOMPUTED_MOLECULES.pk","wb+"))
        self.precomputed = True

    @lru_cache(maxsize=5000)
    def _transform(self, smile):
        tens = torch.from_numpy(self._dc_featurizer.featurize([smile])).squeeze().float()
        if self.use_cuda:
            tens = tens.cuda()

        return tens

    def __call__(self, smile):
        if self.precomputed:
            return self.mol_embs[smile]
        else:
            return self._transform(smile)