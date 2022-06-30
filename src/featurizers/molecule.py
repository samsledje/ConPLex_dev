import torch
import deepchem as dc
import logging as logg

from pathlib import Path
from .base import Featurizer

# class Mol2VecFeaturizer(Featurizer):
#     def __init__(self,
#                  radius: int = 1,
#                  save_dir: Path = Path().absolute()
#                 ):
#         super().__init__("Mol2Vec", 300)

#         self._featurizer = dc.feat.Mol2VecFingerprint(
#             radius = radius,
#         )

#     def _transform(self, smile: str) -> torch.Tensor:
#         feats = torch.from_numpy(self._featurizer(smile)).squeeze().float()
#         return feats


class MorganFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 2048,
        radius: int = 2,
        save_dir: Path = Path().absolute(),
    ):
        super().__init__("Morgan", shape, save_dir)

        self._featurizer = dc.feat.CircularFingerprint(
            radius=radius,
            size=shape,
        )

    def _transform(self, smile: str) -> torch.Tensor:
        feats = torch.from_numpy(self._featurizer(smile)).squeeze().float()
        if feats.shape[0] != self.shape:
            logg.warning("Failed to featurize: appending zero vector")
            feats = torch.zeros(self.shape)
        return feats
