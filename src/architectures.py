import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dscript
import os
import pickle as pk
from types import SimpleNamespace
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
from .utils import get_logger

# from dpatch import PB_Embed
from torch.nn.utils.rnn import pad_sequence

logg = get_logger()

#################################
# Latent Space Distance Metrics #
#################################


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

#######################
# Model Architectures #
#######################


class SimpleCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class SimpleCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=100,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class GoldmanCPI(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=100,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        self.last_linear = nn.Sequential(nn.Linear(latent_dimension, 1))

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)
        output = torch.einsum("bd,bd->bd", drug_projection, target_projection)
        distance = self.last_linear(output)
        return distance

    def classify(self, drug, target):
        distance = self.regress(drug, target)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class AffinityCoembedInner(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.mol_projector[0].weight)

        print(self.mol_projector[0].weight)

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.prot_projector[0].weight)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        print(mol_proj)
        print(prot_proj)
        y = torch.bmm(
            mol_proj.view(-1, 1, self.latent_size),
            prot_proj.view(-1, self.latent_size, 1),
        ).squeeze()
        print("HERE")
        print(y)
        return y
        # return torch.inner(mol_proj, prot_proj).squeeze()


class CosineBatchNorm(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, self.latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, self.latent_size),
            latent_activation(),
        )

        self.mol_norm = nn.BatchNorm1d(self.latent_size)
        self.prot_norm = nn.BatchNorm1d(self.latent_size)

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_norm(self.mol_projector(mol_emb))
        prot_proj = self.prot_norm(self.prot_projector(prot_emb))

        return self.activator(mol_proj, prot_proj)


class LSTMCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        lstm_layers=3,
        lstm_dim=256,
        latent_size=256,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.rnn = nn.LSTM(
            self.prot_emb_size,
            lstm_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(2 * lstm_layers * lstm_dim, latent_size), nn.ReLU()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)

        outp, (h_out, _) = self.rnn(prot_emb)
        prot_hidden = h_out.permute(1, 0, 2).reshape(outp.shape[0], -1)
        prot_proj = self.prot_projector(prot_hidden)

        return self.activator(mol_proj, prot_proj)


class DeepCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        hidden_size=4096,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, hidden_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
            nn.Linear(hidden_size, latent_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class SimpleConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        hidden_dim_1=512,
        hidden_dim_2=256,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.fc1 = nn.Sequential(
            nn.Linear(mol_emb_size + prot_emb_size, hidden_dim_1), activation()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2), activation()
        )
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim_2, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc3(self.fc2(self.fc1(cat_emb))).squeeze()


class SeparateConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric=None,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.fc = nn.Sequential(nn.Linear(2 * latent_size, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


class AffinityEmbedConcat(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )

        self.fc = nn.Linear(2 * latent_size, 1)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


SimplePLMModel = AffinityEmbedConcat


class AffinityConcatLinear(nn.Module):
    def __init__(
        self,
        mol_emb_size,
        prot_emb_size,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.fc = nn.Linear(mol_emb_size + prot_emb_size, 1)

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc(cat_emb).squeeze()
