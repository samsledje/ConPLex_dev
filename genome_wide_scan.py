import os
import sys
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from Bio import SeqIO
from rdkit import Chem
from rdkit.Chem import AllChem

from src.architectures import SimpleCoembedding
from src.featurizers import MorganFeaturizer, ProtBertFeaturizer

# File Paths and Constants
DEVICE = torch.device("cuda:0")
MODEL_STATE_DICT = "/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/best_models/production_bdb_best_model.pt"

SEQUENCE_PATH = "/afs/csail.mit.edu/u/s/samsl/Work/databases/STRING/homo.sapiens/50_800_SEQUENCES_NONRED.fasta"

# DRUG_PATH = "/afs/csail.mit.edu/u/s/samsl/Work/databases/DrugBank/open_structures.sdf"
DRUG_PATH = (
    "/afs/csail.mit.edu/u/s/samsl/Work/databases/ChEMBL/ChEMBL_molecules.sdf"
)
# PROP_NAME = "DRUGBANK_ID"
PROP_NAME = "chembl_id"


CACHE_DIR = "/afs/csail.mit.edu/u/s/samsl/Work/databases/CONPlex_LargeScale/"
RESULTS_PATH = (
    "/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/large_scale_run.tsv"
)


BATCH_SIZE = 2048

# Initialize Model
drug_featurizer = MorganFeaturizer(save_dir=CACHE_DIR).to(DEVICE)
target_featurizer = ProtBertFeaturizer(save_dir=CACHE_DIR).to(DEVICE)

model = SimpleCoembedding(
    drug_featurizer.shape,
    target_featurizer.shape,
    latent_dimension=1024,
    latent_distance="Cosine",
    classify=True,
)

model.load_state_dict(torch.load(MODEL_STATE_DICT))
model = model.to(DEVICE)

drugs = list(Chem.SDMolSupplier(DRUG_PATH))
targets = list(SeqIO.parse(SEQUENCE_PATH, "fasta"))

drug_str = {}
for mol in tqdm(drugs, desc="Loading SMILES"):
    if mol is not None:
        dbid = mol.GetProp(PROP_NAME)
        smile = Chem.MolToSmiles(mol)
        drug_str[dbid] = smile

target_str = {}
for rec in tqdm(targets, desc="Loading Seqs"):
    if rec is not None:
        protid = rec.name
        seq = str(rec.seq)
        target_str[protid] = seq

drug_featurizer.preload(drug_str.values())
target_featurizer.preload(target_str.values())

results = []


def cosine_sim(model, dfeat, tfeat):
    dproj = model.drug_projector(dfeat)
    tproj = model.target_projector(tfeat)
    cos = torch.nn.CosineSimilarity()
    cos_sim = cos(dproj, tproj).detach().cpu().numpy()
    return cos_sim


with open(RESULTS_PATH, "w+") as f, torch.set_grad_enabled(False):
    for dk, dstr in tqdm(drug_str.items(), total=len(drug_str)):
        dfeat = drug_featurizer(dstr)

        batch_tname = []
        batch_tfeat = []
        for i, (tk, tstr) in tqdm(
            enumerate(target_str.items()), total=len(target_str), leave=False
        ):
            tfeat = target_featurizer(tstr)
            batch_tname.append(tk)
            batch_tfeat.append(tfeat)

            if (i + 1) % BATCH_SIZE == 0:
                batch_dfeat = dfeat.repeat(len(batch_tfeat), 1)
                batch_tfeat = torch.stack(batch_tfeat, 0)
                phat = cosine_sim(model, batch_dfeat, batch_tfeat)
                for (tn, ph) in zip(batch_tname, phat):
                    f.write(f"{dk}\t{tn}\t{ph}\n")
                batch_tname = []
                batch_tfeat = []

        if len(batch_tfeat):
            batch_dfeat = dfeat.repeat(len(batch_tfeat), 1)
            batch_tfeat = torch.stack(batch_tfeat, 0)
            phat = cosine_sim(model, batch_dfeat, batch_tfeat)
            for (tn, ph) in zip(batch_tname, phat):
                f.write(f"{dk}\t{tn}\t{ph}\n")
        f.flush()
