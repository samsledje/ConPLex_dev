import torch
from ..featurizers import (
    NullFeaturizer,
    BeplerBergerFeaturizer,
    ProtBertFeaturizer,
    MorganFeaturizer,
)


def test_featurizer():
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

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    g = ProtBertFeaturizer()
    g.cuda(device)
    g.preload(seqs)

    h = MorganFeaturizer()
    h.cuda(device)
    h.preload(drugs)
