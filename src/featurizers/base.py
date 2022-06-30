from __future__ import annotations
import torch
import h5py
import typing as T
import numpy as np

from tqdm import tqdm
from pathlib import Path
from functools import lru_cache

from ..utils import get_logger

logg = get_logger()

###################
# Base Featurizer #
###################


class Featurizer:
    def __init__(
        self, name: str, shape: int, save_dir: Path = Path().absolute()
    ):
        self._name = name
        self._shape = shape
        self._save_path = save_dir / Path(f"{self._name}_features.h5")

        self._preloaded = False
        self._device = torch.device("cpu")
        self._cuda_registry = {}
        self._on_cuda = False
        self._features = {}

    def __call__(self, seq: str) -> torch.Tensor:
        if seq not in self.features:
            self._features[seq] = self.transform(seq)

        return self._features[seq]

    def _register_cuda(self, k: str, v, f=None):
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
                self._cuda_registry[k] = (v.to(self.device), None)
            else:
                self._cuda_registry[k] = (f(v, self.device), f)
        for k, v in self._features.items():
            self._features[k] = v.to(device)

    @lru_cache(maxsize=5000)
    def transform(self, seq: str) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            feats = self._transform(seq)
            if self._on_cuda:
                feats = feats.cuda()
            return feats

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def path(self) -> Path:
        return self._save_path

    @property
    def features(self) -> dict:
        return self._features

    @property
    def on_cuda(self) -> bool:
        return self._on_cuda

    @property
    def device(self) -> torch.device:
        return self._device

    def cuda(self, device: torch.device) -> Featurizer:
        """
        Perform model computations on CUDA, move saved embeddings to CUDA device
        """
        self._update_device(device)
        self._on_cuda = True
        return self

    def cpu(self) -> Featurizer:
        """
        Perform model computations on CPU, move saved embeddings to CPU
        """
        self._update_device(torch.device("cpu"))
        self._on_cuda = False
        return self

    def write_to_disk(self, seq_list: T.List[str]) -> None:
        with h5py.File(self._save_path, "a") as h5fi:
            logg.info(f"Writing {self.name} features to {self.path}")
            for seq in tqdm(seq_list):
                dset = h5fi.require_dataset(seq, (self._shape,), np.float32)
                feats = self.transform(seq)
                dset[:] = feats.cpu().numpy()

    def preload(self, seq_list: T.List[str]) -> None:
        if not self._save_path.exists():
            self.write_to_disk(seq_list)

        with h5py.File(self._save_path, "r") as h5fi:
            logg.info(f"Preloading {self.name} features from {self.path}")
            for seq in tqdm(seq_list):
                feats = torch.from_numpy(h5fi[seq][:])
                if self._on_cuda:
                    feats = feats.to(self.device)

                self._features[seq] = feats
        self._preloaded = True


###################
# Null and Random #
###################


class NullFeaturizer(Featurizer):
    def __init__(self, shape: int = 1024):
        super().__init__(f"Null{shape}", shape)

    def _transform(self, seq: str) -> torch.Tensor:
        return torch.zeros(self.shape)


class RandomFeaturizer(Featurizer):
    def __init__(self, shape: int = 1024):
        super().__init__(f"Random{shape}", shape)

    def _transform(self, seq: str) -> torch.Tensor:
        return torch.rand(self.shape)
