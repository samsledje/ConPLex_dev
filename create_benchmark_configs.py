from sklearn.model_selection import ParameterGrid
from omegaconf import OmegaConf
import os

wandb_project = "CONPlex_Benchmarking"
os.makedirs(f"./configs/{wandb_project}", exist_ok=True)
os.makedirs(f"./results/{wandb_project}", exist_ok=True)
os.makedirs(f"./best_models/{wandb_project}", exist_ok=True)

tasks = [
    "davis",
    "bindingdb",
    "biosnap",
    "biosnap_prot",
    "biosnap_mol",
    "dti_dg",
]

conplex_defaults = {
    "contrastive_split": "within",
    "drug_featurizer": "MorganFeaturizer",
    "target_featurizer": "ProtBertFeaturizer",
    "model_architecture": "SimpleCoembedding",
    "latent_dimension": 1024,
    "latent_distance": "Cosine",
    "batch_size": 32,
    "contrastive_batch_size": 256,
    "shuffle": True,
    "num_workers": 0,
    "epochs": 50,
    "every_n_val": 1,
    "contrastive": True,
    "lr": 1e-4,
    "lr_t0": 10,
    "clr": 1e-5,
    "clr_t0": 10,
    "margin_fn": "tanh_decay",
    "margin_max": 0.25,
    "margin_t0": 10,
    "verbosity": 3,
    "wandb_proj": wandb_project,
    "wandb_save": False,
    "model_save_dir": f"./best_models/{wandb_project}",
}

goldman_defaults = {
    "contrastive_split": "within",
    "drug_featurizer": "MorganFeaturizer",
    "target_featurizer": "ESMFeaturizer",
    "model_architecture": "GoldmanCPI",
    "latent_dimension": 90,
    "latent_distance": "Cosine",
    "batch_size": 32,
    "contrastive_batch_size": 256,
    "shuffle": True,
    "num_workers": 0,
    "epochs": 100,
    "every_n_val": 1,
    "contrastive": False,
    "lr": 1e-5,
    "lr_t0": 10,
    "clr": 1e-5,
    "clr_t0": 10,
    "margin_fn": "tanh_decay",
    "margin_max": 0.25,
    "margin_t0": 10,
    "verbosity": 3,
    "wandb_proj": wandb_project,
    "wandb_save": False,
    "model_save_dir": f"./best_models/{wandb_project}",
}

N_GPUS = 8
N_REPLICATES = 5
config_files = {}

for i, t in enumerate(tasks):
    for j, r in enumerate(range(N_REPLICATES)):
        param_name = "_".join(["CON-Plex", t, str(r)])
        oc = OmegaConf.structured(conplex_defaults)
        oc.task = t
        oc.device = 0  # ((N_REPLICATES*i) + j) % N_GPUS
        oc.replicate = r
        oc.exp_id = param_name
        oc.log_file = f"results/{wandb_project}/{param_name}_log.txt"

        filename = f"configs/{wandb_project}/config_{param_name}.yaml"
        config_files[param_name] = filename
        OmegaConf.save(config=oc, f=filename)

base_cmd = "python train_DTI.py --config {} --exp-id {} "
list_file = f"configs/{wandb_project}/benchmark_sweep_list.txt"
with open(list_file, "w+") as f:
    for pname, fi in config_files.items():
        cmd = base_cmd.format(fi, pname)
        f.write(f"{cmd}\n")

bash_file = f"configs/{wandb_project}/benchmark_sweep_run.sh"
with open(bash_file, "w+") as f:
    f.write(
        f"simple_gpu_scheduler --gpus {' '.join([str(i) for i in range(N_GPUS)])} < {list_file}"
    )
