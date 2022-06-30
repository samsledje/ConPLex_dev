from sklearn.model_selection import ParameterGrid
from omegaconf import OmegaConf

wandb_project = "DTI_Benchmarking_2022-06-29"

tasks = [
    "davis",
    "bindingdb",
    "biosnap",
    "biosnap_prot",
    "biosnap_mol",
    "dti_dg",
]

drug_featurizers = [
    "MorganFeaturizer",
    # 'Mol2VecFeaturizer',
    # 'MolRFeaturizer',
    # 'GEMFeaturizer'
]

target_featurizers = [
    "BeplerBergerFeaturizer",
    "ESMFeaturizer",
    "ProseFeaturizer",
    "ProtBertFeaturizer",
    "ProtT5XLUniref50Featurizer",
]

param_grid = ParameterGrid(
    {
        "task": tasks,
        "drug_featurizer": drug_featurizers,
        "target_featurizer": target_featurizers,
        "replicate": [1, 2, 3, 4, 5],
    }
)

defaults = {
    "contrastive_split": "within",
    "model_architecture": "SimpleCoembedding",
    "latent_dimension": 1024,
    "latent_distance": "Cosine",
    "batch_size": 32,
    "shuffle": True,
    "num_workers": 0,
    "model_save_dir": "./best_models",
    "epochs": 50,
    "every_n_val": 1,
    "lr": 1e-4,
    "clr": 1e-5,
    "verbosity": 3,
}

N_GPUS = 8
config_files = {}

param_sets = list(param_grid)
param_sets.sort(key=lambda x: x["task"], reverse=True)

for i, param in enumerate(param_sets):
    param_name = "_".join([f"{k}:{v}" for k, v in param.items()])
    oc = OmegaConf.structured(param)
    oc.device = i % N_GPUS
    oc.update(defaults)

    filename = f"configs/config_{param_name}.yaml"
    config_files[param_name] = filename
    OmegaConf.save(config=oc, f=filename)

base_cmd = "python train_DTI.py --wandb-proj {} --exp-id {} --config {}"
list_file = "configs/benchmark_sweep_list.txt"
with open(list_file, "w+") as f:
    for key, fi in config_files.items():
        cmd = base_cmd.format(wandb_project, key, fi)
        f.write(f"{cmd}\n")

bash_file = "configs/benchmark_sweep_run.sh"
with open(bash_file, "w+") as f:
    f.write(
        f"simple_gpu_scheduler --gpus {' '.join([str(i) for i in range(N_GPUS)])} < {list_file}"
    )
