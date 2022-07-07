import torch
import deepchem as dc
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from pathlib import Path
from .base import Featurizer
from ..utils import get_logger, canonicalize

logg = get_logger()

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

        self._radius = radius

    def smiles_to_morgan(self, smile: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        try:
            smile = canonicalize(smile)
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, self._radius, nBits=self.shape
            )
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except Exception as e:
            logg.error(
                f"rdkit not found this smiles for morgan: {smile} convert to all 0 features"
            )
            logg.error(e)
            features = np.zeros((self.shape,))
        return features

    def _transform(self, smile: str) -> torch.Tensor:
        # feats = torch.from_numpy(self._featurizer(smile)).squeeze().float()
        feats = (
            torch.from_numpy(self.smiles_to_morgan(smile)).squeeze().float()
        )
        if feats.shape[0] != self.shape:
            logg.warning("Failed to featurize: appending zero vector")
            feats = torch.zeros(self.shape)
        return feats
