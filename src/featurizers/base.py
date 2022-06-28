import torch
import h5py
import typing as T
import numpy as np

from tqdm import tqdm
from pathlib import Path
from functools import lru_cache

###################
# Base Featurizer #
###################

class Featurizer:
    def __init__(self,
                 name: str,
                 shape: int,
                 save_dir: Path = Path().absolute()
                ):
        self._name = name
        self._shape = shape
        self._save_path = save_dir / Path(f"{self._name}_features.h5")
        
        self._precomputed = False
        self._device = torch.device("cpu")
        self._cuda_registry = {}
        self._on_cuda = False
        self._features = {}
                                 
    def __call__(self, seq: str) -> torch.Tensor:
        if seq not in self.features:
            self._features[seq] = self.transform(seq)
        
        return self._features[seq]
    
    def _register_cuda(self, k: str, v, f = None):
        """
        Register an object as capable of being moved to a CUDA device
        """
        self._cuda_registry[k] = (v, f)

    def _transform(self, seq: str) -> torch.Tensor:
        raise NotImplementedError
            
    def _update_device(self, device: torch.device):
        self._device = device
        for k, (v, f) in self._cuda_registry.items():
            if f is None:
                self._cuda_registry[k] = (v.to(device), None)
            else:
                self._cuda_registry[k] = (f(device), f)
        for k, v in self._features.items():
            self._features[k] = v.to(device)
            
    @lru_cache(maxsize=5000)
    def transform(self, seq: str) -> torch.Tensor:
        feats = self._transform(seq)
        if self._on_cuda:
            feats = feats.cuda()
        return feats
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def shape(self) -> str:
        return self._shape 
        
    @property
    def path(self) -> str:
        return self._save_path
    
    @property
    def features(self) -> dict:
        return self._features
    
    @property
    def on_cuda(self) -> bool:
        return self._on_cuda
    
    def cuda(self, device: torch.device):
        """
        Perform model computations on CUDA, move saved embeddings to CUDA device
        """
        self._update_device(device)
        self._on_cuda = True
        return self
    
    def cpu(self):
        """
        Perform model computations on CPU, move saved embeddings to CPU
        """
        self._update_device(torch.device('cpu'))
        self._on_cuda = False
        return self
        
    def precompute(self, seq_list: T.List[str]) -> None:
        with h5py.File(self._save_path,"a") as h5fi:
            for seq in tqdm(seq_list, desc = f"Pre-computing {self._name} features"):
                if seq in h5fi.keys():
                    feats = torch.from_numpy(h5fi[seq][:])
                    if self._on_cuda:
                        feats.to(self._device)
                else:
                    dset = h5fi.require_dataset(
                        seq, (self._shape,), np.float32
                    )
                    feats = self._transform(seq)
                    dset[:] = feats
                
                if self._on_cuda:
                    feats = feats.cuda()
                    
                self._features[seq] = feats
        self._precomputed = True
        


###################
# Null and Random #
###################

class NullFeaturizer(Featurizer):
    def __init__(self,
                 shape: int = 1024
                ):
        super().__init__(f"Null{shape}", shape)
        
    def _transform(self, seq: str) -> torch.Tensor:
        return torch.zeros(self._shape)
    
class RandomFeaturizer(Featurizer):
    def __init__(self,
                 shape: int = 1024
                ):
        super().__init__(f"Random{shape}", shape)