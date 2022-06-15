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

# Parse Arguments
parser = ArgumentParser()
parser.add_argument('target', help='DUDe Target')
parser.add_argument('--database',
                    default='/afs/csail.mit.edu/u/s/samsl/Work/databases/DUDe',
                    help='Location of DUDe database'
                   )
parser.add_argument('--model',
                    default='/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/best_models/bindingdb_morgan_protbert_best_model.sav',
                    help='Path to saved DTI model'
                   )
parser.add_argument('--outdir',
                    default='/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/img',
                    help='Location to write outputs'
                   )

args = parser.parse_args()
target = args.target
outdir = args.outdir
database_path = args.database
model_path = args.model
model_name = model_path.split('/')[-1].split('.')[0]

# Load Molecules
try:
    actives = Chem.SDMolSupplier(f'{database_path}/{target}/actives_final.sdf')
    decoys = Chem.SDMolSupplier(f'{database_path}/{target}/decoys_final.sdf')
except OSError:
    raise OSError(f"Need to first gunzip {database_path}/{target}/actives_final.sdf.gz and {database_path}/{target}/decoys_final.sdf")
    sys.exit(1)

# Load Target Sequence
parser = PDBParser()
structure = parser.get_structure(target, f'{database_path}/{target}/receptor.pdb')
seq = list(AtomIterator(target, structure))[0]

# Load Model
from architectures import SimplePLMModel, SimpleCosine
from mol_feats import Morgan_f
from prot_feats import ProtBert_f
mol_f = Morgan_f()
prot_f = ProtBert_f()
model = torch.load(model_path).cuda()

# Project target and molecules
active_projections = []
decoy_projections = []

with torch.set_grad_enabled(False):
    for m in tqdm(actives):
        m_emb = mol_f(Chem.MolToSmiles(m)).cuda()
        m_proj = model.mol_projector(m_emb).detach().cpu().numpy()
        active_projections.append(m_proj)
    for m in tqdm(decoys):
        try:
            m_emb = mol_f(Chem.MolToSmiles(m)).cuda()
            m_proj = model.mol_projector(m_emb).detach().cpu().numpy()
            decoy_projections.append(m_proj)
        except Exception as e:
            continue
    seq_proj = model.prot_projector(prot_f(str(seq)).cuda()).cpu().numpy()

# Evaluate Distributions
cosine_sim = CosineSimilarity(dim=0)
active_scores = []
decoy_scores = []

pProj = torch.from_numpy(seq_proj).cuda()

with torch.set_grad_enabled(False):
    for mProj in tqdm(active_projections):
        affin = cosine_sim(pProj,torch.from_numpy(mProj).cuda()).cpu().numpy()
        active_scores.append(float(affin))
    for mProj in tqdm(decoy_projections):
        affin = cosine_sim(pProj,torch.from_numpy(mProj).cuda()).cpu().numpy()
        decoy_scores.append(float(affin))

df = pd.DataFrame({"scores":active_scores + decoy_scores,"label":(['active']*len(active_scores))+(['decoy']*len(decoy_scores))})
stat, pvalue = scipy.stats.ttest_ind(df[df['label'] == 'active']['scores'], df[df['label'] == 'decoy']['scores'],alternative='greater')
print(f'T stat={stat}, p={pvalue}')
with open(f'{outdir}/DUDe_{target}_{model_name}_pval.txt','w+') as f:
    f.write(f"{target}\t{stat}\t{pvalue}\n")
df.to_csv(f'{outdir}/DUDe_{target}_{model_name}_scores.csv')

sns.violinplot(data=df, x='label', y='scores')
plt.title(f'{target} Predicted Scores (p={pvalue})')
plt.savefig(f'{outdir}/DUDe_{target}_{model_name}_violinplot.png',bbox_inches='tight')
# plt.show()

sns.displot(data=df,x='scores',hue='label')
plt.title(f'{target} Predicted Scores (p={pvalue})')
plt.savefig(f'{outdir}/DUDe_{target}_{model_name}_displot.png',bbox_inches='tight')
# plt.show()

# Plot TSNE of projections
all_projections = np.concatenate([active_projections,decoy_projections,[seq_proj]],axis=0)
project_tsne = TSNE(metric="cosine",n_jobs=32).fit_transform(all_projections)

hue = ['Active']*len(active_projections)+['Decoy']*len(decoy_projections)+['Target']
size = [15]*len(active_projections)+[15]*len(decoy_projections)+[100]
plt.figure(figsize=(15, 15), dpi=80)
sns.scatterplot(x=project_tsne[:,0],
                y=project_tsne[:,1],
                hue=hue,
                s=size
               )
plt.title(f'{target} T-SNE')
plt.savefig(f'{outdir}/DUDe_{target}_{model_name}_tsne.png',bbox_inches='tight')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
sns.despine()
# plt.show()


