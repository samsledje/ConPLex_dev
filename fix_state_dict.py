import sys
import torch

from pathlib import Path
from src.architectures import SimpleCoembedding


def rename_state_dict(x):
    d = {}
    d["drug_projector.0.weight"] = x["mol_projector.0.weight"]
    d["drug_projector.0.bias"] = x["mol_projector.0.bias"]
    d["target_projector.0.weight"] = x["prot_projector.0.weight"]
    d["target_projector.0.bias"] = x["prot_projector.0.bias"]
    return d


model = SimpleCoembedding(
    2048,
    1024,
    latent_dimension=1024,
    latent_distance="Cosine",
    classify=True,
)

model_path = Path(sys.argv[1])
path_pt = model_path.with_suffix(".pt")
print(model_path, "to", path_pt)

old_model = torch.load(model_path)
sd = old_model.state_dict()
sd_new = rename_state_dict(sd)

try:
    y = model.load_state_dict(sd_new)
    print(y)
    torch.save(sd_new, path_pt)
except Exception as e:
    print(e)
