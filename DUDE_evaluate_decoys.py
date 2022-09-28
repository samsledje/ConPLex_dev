import os
import sys

import numpy as np
import pandas as pd
import scipy
import torch

from tqdm import tqdm
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns

from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.SeqIO.PdbIO import AtomIterator
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from torch.nn import CosineSimilarity
from scipy.spatial.distance import cosine

from src.utils import get_logger

logg = get_logger()

# Parse Arguments
parser = ArgumentParser()
parser.add_argument("target", help="DUDe Target")
parser.add_argument(
    "--database",
    default="/afs/csail.mit.edu/u/s/samsl/Work/databases/DUDe",
    help="Location of DUDe database",
)
parser.add_argument(
    "--model",
    default="/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/best_models/bindingdb_morgan_protbert_best_model.sav",
    help="Path to saved DTI model",
)
parser.add_argument(
    "--outdir",
    default="/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/img",
    help="Location to write outputs",
)

args = parser.parse_args()
target = args.target
outdir = args.outdir
database_path = args.database
model_path = args.model
model_name = model_path.split("/")[-1].split(".")[0]

device = torch.device("cuda:0")

# Load Molecules
try:
    actives = Chem.SDMolSupplier(f"{database_path}/{target}/actives_final.sdf")
    decoys = Chem.SDMolSupplier(f"{database_path}/{target}/decoys_final.sdf")
except OSError:
    raise OSError(
        f"Need to first gunzip {database_path}/{target}/actives_final.sdf.gz and {database_path}/{target}/decoys_final.sdf"
    )
    sys.exit(1)

# Load Target Sequence
parser = PDBParser()
structure = parser.get_structure(
    target, f"{database_path}/{target}/receptor.pdb"
)
seq = list(AtomIterator(target, structure))[0]
print(seq)

# Load Model
from src.architectures import SimpleCoembedding, GoldmanCPI
from src.featurizers import MorganFeaturizer, ProtBertFeaturizer, ESMFeaturizer

drug_featurizer = MorganFeaturizer().to(device)
target_featurizer = ProtBertFeaturizer().to(device)
# target_featurizer = ESMFeaturizer().to(device)

state_dict = torch.load(model_path)

model = SimpleCoembedding(
    drug_featurizer.shape,
    target_featurizer.shape,
    latent_dimension=1024,
    latent_distance="Cosine",
    classify=True,
).to(device)


# bash scripts/CMD_BENCHMARK_DUDE_WITHINTYPE.sh best_models/goldman_bindingdb_best_model.pt results/goldmanBDB_DUDEwithin
# model = GoldmanCPI(
#         drug_featurizer.shape,
#         target_featurizer.shape,
#         latent_dimension=100,
#         classify=True,
#     ).to(device)

model = torch.load(model_path)
# model.load_state_dict(state_dict)
model = model.cuda().eval()

# Project target and molecules
active_projections = []
decoy_projections = []

with torch.set_grad_enabled(False):
    for m in tqdm(actives):
        m_emb = drug_featurizer(Chem.MolToSmiles(m)).cuda()
        # m_proj = model.drug_projector(m_emb).detach().cpu().numpy()
        m_proj = model.mol_projector(m_emb).detach().cpu().numpy()
        active_projections.append(m_proj)
    for m in tqdm(decoys):
        try:
            m_emb = drug_featurizer(Chem.MolToSmiles(m)).cuda()
            # m_proj = model.drug_projector(m_emb).detach().cpu().numpy()
            m_proj = model.mol_projector(m_emb).detach().cpu().numpy()
            decoy_projections.append(m_proj)
        except Exception as e:
            logg.error(e)
    try:
        seq_proj = (
            # model.target_projector(target_featurizer(str(seq.seq)).cuda())
            model.prot_projector(target_featurizer(str(seq.seq)).cuda())
            .cpu()
            .numpy()
        )
    except KeyError as e:
        logg.error(e)
        logg.debug(str(seq))
        sys.exit(1)

# Evaluate Distributions
cosine_sim = CosineSimilarity(dim=0)
active_scores = []
decoy_scores = []

pProj = torch.from_numpy(seq_proj).cuda()

with torch.set_grad_enabled(False):
    for mProj in tqdm(active_projections):
        affin = cosine_sim(pProj, torch.from_numpy(mProj).cuda()).cpu().numpy()
        active_scores.append(float(affin))
    for mProj in tqdm(decoy_projections):
        affin = cosine_sim(pProj, torch.from_numpy(mProj).cuda()).cpu().numpy()
        decoy_scores.append(float(affin))

df = pd.DataFrame(
    {
        "scores": active_scores + decoy_scores,
        "label": (["Active"] * len(active_scores))
        + (["Decoy"] * len(decoy_scores)),
    }
)
stat, pvalue = scipy.stats.ttest_ind(
    df[df["label"] == "Active"]["scores"],
    df[df["label"] == "Decoy"]["scores"],
    alternative="greater",
)
print(f"T stat={stat}, p={pvalue}")
with open(f"{outdir}/DUDe_{target}_{model_name}_pval.txt", "w+") as f:
    f.write(f"{target}\t{stat}\t{pvalue}\n")
df.to_csv(f"{outdir}/DUDe_{target}_{model_name}_scores.csv")

sns.set(style="whitegrid", font_scale=3)
plt.figure(figsize=(15, 15), dpi=100)
sns.violinplot(data=df, x="label", y="scores")
plt.title(f"{target} Predicted Scores (p={pvalue})")
plt.xlabel("Molecule")
plt.ylabel("Predicted Score")
plt.savefig(
    f"{outdir}/DUDe_{target}_{model_name}_violinplot.svg", bbox_inches="tight"
)
plt.savefig(
    f"{outdir}/DUDe_{target}_{model_name}_violinplot.png", bbox_inches="tight"
)
# plt.show()

sns.displot(data=df, x="scores", hue="label")
plt.title(f"{target} Predicted Scores (p={pvalue})")
plt.savefig(
    f"{outdir}/DUDe_{target}_{model_name}_displot.svg", bbox_inches="tight"
)
# plt.show()

# Plot TSNE of projections
all_projections = np.concatenate(
    [active_projections, decoy_projections, [seq_proj]], axis=0
)
project_tsne = TSNE(
    metric="cosine", n_jobs=32, random_state=61998
).fit_transform(all_projections)

hue = (
    ["Active"] * len(active_projections)
    + ["Decoy"] * len(decoy_projections)
    + ["Target"]
)
size = [30] * len(active_projections) + [30] * len(decoy_projections) + [1000]

sns.set(style="whitegrid", font_scale=3)
plt.figure(figsize=(15, 15), dpi=100)
sns.scatterplot(
    x=project_tsne[:, 0], y=project_tsne[:, 1], hue=hue, s=size, legend=False
)
# plt.title(f"{target} T-SNE")
plt.xlabel("T-SNE 1")
plt.ylabel("T-SNE 2")
# plt.legend()
sns.despine()
plt.savefig(
    f"{outdir}/DUDe_{target}_{model_name}_tsne.svg", bbox_inches="tight"
)
plt.savefig(
    f"{outdir}/DUDe_{target}_{model_name}_tsne.png", bbox_inches="tight"
)
# plt.show()
