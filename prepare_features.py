import torch

from argparse import ArgumentParser

from src import featurizers
from src.utils import config_logger
from src.data import get_task_dir, DTIDataModule, DUDEDataModule, TDCDataModule

parser = ArgumentParser(description="Write DTI features to disk.")

parser.add_argument(
    "--task",
    required=True,
    choices=[
        "biosnap",
        "biosnap_prot",
        "biosnap_mol",
        "bindingdb",
        "davis",
        "dti_dg",
        "dude",
    ],
    type=str,
    help="Task to generate features for",
)
parser.add_argument(
    "--drug-featurizer",
    type=str,
    help="Drug featurizer",
    dest="drug_featurizer",
)
parser.add_argument(
    "--target-featurizer",
    type=str,
    help="Target featurizer",
    dest="target_featurizer",
)
parser.add_argument(
    "--d", "--device", type=int, help="CUDA device", dest="device", default=-1
)


def main():
    # Get configuration
    config = parser.parse_args()

    # Logging
    logg = config_logger(
        None,
        "%(asctime)s [%(levelname)s] %(message)s",
        3,
        use_stdout=True,
    )
    logg.propagate = False

    # Set CUDA device
    device_no = config.device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    logg.info(f"Using CUDA device {device}")

    # Prepare DatamModule
    logg.info("Preparing DataModule")
    task_dir = get_task_dir(config.task)

    drug_featurizer = getattr(featurizers, config.drug_featurizer)(
        save_dir=task_dir
    )
    target_featurizer = getattr(featurizers, config.target_featurizer)(
        save_dir=task_dir
    )

    if config.task == "dti_dg":
        datamodule = TDCDataModule(
            task_dir, drug_featurizer, target_featurizer, device=device
        )
    elif config.task == "dude":
        datamodule = DUDEDataModule(
            "within",
            drug_featurizer,
            target_featurizer,
            device=device,
        )
    else:
        datamodule = DTIDataModule(
            task_dir, drug_featurizer, target_featurizer, device=device
        )
    datamodule.prepare_data()


if __name__ == "__main__":
    main()
