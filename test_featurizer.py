import torch
from src.featurizers import (
    NullFeaturizer,
    BeplerBergerFeaturizer,
    ProtBertFeaturizer,
    Mol2VecFeaturizer,
    MorganFeaturizer,
)

seqs = [
    "MTQMSQVQELFHEAAQQDALAQPQPWWKTQLFMWEPVLFGTWDGVF",
    "MAANSTSDLHTPGTQLSVADIIVITVYFALNVAVGIWSSCRASRNT",
    "MEERAQHCLSRLLDNSALKQQELPIHRLYFTARRVLFVFFATGIFC",
]

drugs = [
    "Cc1ccc(CNS(=O)(=O)c2ccc(s2)S(N)(=O)=O)cc1",
    "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N",
    "CC(C)(C)C1=CC(=NO1)NC(=O)NC2=CC=C(C=C2)C3=CN4C5=C(C=C(C=C5)OCCN6CCOCC6)SC4=N3",
]

device = torch.device(0)

# f = NullFeaturizer()
# f.precompute(seqs)

g = BeplerBergerFeaturizer()
# g = ProtBertFeaturizer()
g.cuda(device)
g.precompute(seqs)

h = Mol2VecFeaturizer()
h.cuda(device)
# h = MorganFeaturizer()
h.precompute(drugs)
