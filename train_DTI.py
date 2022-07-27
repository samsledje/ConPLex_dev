import copy
from time import time
import os
import sys
import numpy as np
import pandas as pd
import torch
import json
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import typing as T
import torchmetrics

from argparse import ArgumentParser

import wandb
from omegaconf import OmegaConf
from pathlib import Path

from src import featurizers
from src import architectures as model_types
from src.data import (
    get_task_dir,
    DTIDataModule,
    TDCDataModule,
    DUDEDataModule,
    EnzPredDataModule,
)
from src.utils import (
    set_random_seed,
    config_logger,
    get_logger,
    get_featurizer,
    sigmoid_cosine_distance_p,
)

logg = get_logger()

parser = ArgumentParser(description="PLM_DTI Training.")
parser.add_argument(
    "--exp-id", required=True, help="Experiment ID", dest="experiment_id"
)
parser.add_argument("--config", required=True, help="YAML config file")

parser.add_argument(
    "--wandb-proj",
    help="Weights and Biases Project",
    dest="wandb_proj",
)
parser.add_argument(
    "--task",
    choices=[
        "biosnap",
        "bindingdb",
        "davis",
        "biosnap_prot",
        "biosnap_mol",
        "dti_dg",
    ],
    type=str,
    help="Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol.",
)
parser.add_argument("--contrastive", action="store_true")

parser.add_argument(
    "--drug-featurizer", help="Drug featurizer", dest="drug_featurizer"
)
parser.add_argument(
    "--target-featurizer", help="Target featurizer", dest="target_featurizer"
)
parser.add_argument(
    "--distance-metric",
    help="Distance in embedding space to supervise with",
    dest="distance_metric",
)
parser.add_argument("--epochs", type=int, help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", type=int, help="batch size")
parser.add_argument(
    "--lr",
    "--learning-rate",
    type=float,
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--clr", type=float, help="initial learning rate", dest="clr"
)
parser.add_argument(
    "--r", "--replicate", type=int, help="Replicate", dest="replicate"
)
parser.add_argument(
    "--d", "--device", type=int, help="CUDA device", dest="device"
)
parser.add_argument(
    "--verbosity", type=int, help="Level at which to log", dest="verbosity"
)
parser.add_argument(
    "--checkpoint", default=None, help="Model weights to start from"
)


def test(model, data_generator, metrics, device=None, classify=True):

    if device is None:
        device = torch.device("cpu")

    metric_dict = {}

    for k, met_class in metrics.items():
        met_instance = met_class()
        met_instance.to(device)
        met_instance.reset()
        metric_dict[k] = met_instance

    model.eval()

    for i, batch in tqdm(enumerate(data_generator), total=len(data_generator)):

        pred, label = step(model, batch, device)
        if classify:
            label = label.int()
        else:
            label = label.float()

        for _, met_instance in metric_dict.items():
            met_instance(pred, label)

    results = {}
    for (k, met_instance) in metric_dict.items():
        res = met_instance.compute()
        results[k] = res

    for met_instance in metric_dict.values():
        met_instance.to("cpu")

    return results


def step(model, batch, device=None):

    if device is None:
        device = torch.device("cpu")

    drug, target, label = batch

    pred = model(drug.to(device), target.to(device))
    label = Variable(torch.from_numpy(np.array(label)).float()).to(device)
    return pred, label


def contrastive_step(model, batch, device=None):

    if device is None:
        device = torch.device("cpu")

    anchor, positive, negative = batch

    anchor_projection = model.target_projector(anchor.to(device))
    positive_projection = model.drug_projector(positive.to(device))
    negative_projection = model.drug_projector(negative.to(device))

    return anchor_projection, positive_projection, negative_projection


def wandb_log(m, do_wandb=True):
    if do_wandb:
        wandb.log(m)


def main():
    # Get configuration
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(arg_overrides)

    save_dir = config.get("model_save_dir", ".")
    os.makedirs(save_dir, exist_ok=True)

    # Logging
    if "log_file" not in config:
        config.log_file = None
    config_logger(
        config.log_file,
        "%(asctime)s [%(levelname)s] %(message)s",
        config.verbosity,
        use_stdout=True,
    )

    # Set CUDA device
    device_no = config.device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    logg.info(f"Using CUDA device {device}")

    # Set random state
    logg.debug(f"Setting random state {config.replicate}")
    set_random_seed(config.replicate)

    # Load DataModule
    logg.info("Preparing DataModule")
    task_dir = get_task_dir(config.task)

    drug_featurizer = get_featurizer(config.drug_featurizer, save_dir=task_dir)
    target_featurizer = get_featurizer(
        config.target_featurizer, save_dir=task_dir
    )

    if config.task == "dti_dg":
        config.classify = False
        config.watch_metric = "val/pcc"
        datamodule = TDCDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
    elif config.task in EnzPredDataModule.dataset_list():
        config.classify = True
        config.watch_metric = "val/aupr"
        datamodule = EnzPredDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
    else:
        config.classify = True
        config.watch_metric = "val/aupr"
        datamodule = DTIDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
    datamodule.prepare_data()
    datamodule.setup()

    # Load DataLoaders
    logg.info("Getting DataLoaders")
    training_generator = datamodule.train_dataloader()
    validation_generator = datamodule.val_dataloader()
    testing_generator = datamodule.test_dataloader()

    if config.contrastive:
        logg.info("Loading contrastive data (DUDE)")
        dude_drug_featurizer = get_featurizer(
            config.drug_featurizer, save_dir=get_task_dir("DUDe")
        )
        dude_target_featurizer = get_featurizer(
            config.target_featurizer, save_dir=get_task_dir("DUDe")
        )

        contrastive_datamodule = DUDEDataModule(
            config.contrastive_split,
            dude_drug_featurizer,
            dude_target_featurizer,
            device=device,
            batch_size=config.contrastive_batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
        contrastive_datamodule.prepare_data()
        contrastive_datamodule.setup(stage="fit")
        contrastive_generator = contrastive_datamodule.train_dataloader()

        logg.debug(next(contrastive_generator))
        sys.exit(1)

    config.drug_shape = drug_featurizer.shape
    config.target_shape = target_featurizer.shape

    # Model
    logg.info("Initializing model")
    model = getattr(model_types, config.model_architecture)(
        config.drug_shape,
        config.target_shape,
        latent_dimension=config.latent_dimension,
        latent_distance=config.latent_distance,
        classify=config.classify,
    )
    if "checkpoint" in config:
        state_dict = torch.load(config.checkpoint)
        model.load_state_dict(state_dict)

    model = model.to(device)
    logg.info(model)

    # Optimizers
    logg.info("Initializing optimizers")
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.contrastive:
        opt_contrastive = torch.optim.Adam(model.parameters(), lr=config.clr)

    # Metrics
    logg.info("Initializing metrics")
    max_metric = 0
    # max_epoch = 0
    triplet_margin = 0
    model_max = copy.deepcopy(model)

    if config.task == "dti_dg":
        loss_fct = torch.nn.MSELoss()
        val_metrics = {
            "val/mse": torchmetrics.MeanSquaredError,
            "val/pcc": torchmetrics.PearsonCorrCoef,
        }

        test_metrics = {
            "test/mse": torchmetrics.MeanSquaredError,
            "test/pcc": torchmetrics.PearsonCorrCoef,
        }
    else:
        loss_fct = torch.nn.BCELoss()
        val_metrics = {
            "val/aupr": torchmetrics.AveragePrecision,
            "val/auroc": torchmetrics.AUROC,
            "val/f1": torchmetrics.F1Score,
        }

        test_metrics = {
            "test/aupr": torchmetrics.AveragePrecision,
            "test/auroc": torchmetrics.AUROC,
            "test/f1": torchmetrics.F1Score,
        }

    # Initialize wandb
    do_wandb = "wandb_proj" in config
    if do_wandb:
        logg.debug(f"Initializing wandb project {config.wandb_proj}")
        wandb.init(
            project=config.wandb_proj,
            name=config.experiment_id,
            config=dict(config),
        )
        wandb.watch(model, log_freq=100)
    logg.info("Config:")
    logg.info(json.dumps(dict(config), indent=4))

    logg.info("Beginning Training")

    torch.backends.cudnn.benchmark = True

    start_time = time()
    for epo in range(config.epochs):
        model.train()
        epoch_time_start = time()

        for i, batch in tqdm(
            enumerate(training_generator), total=len(training_generator)
        ):

            pred, label = step(model, batch, device)

            loss = loss_fct(pred, label)

            wandb_log(
                {"train/loss": loss},
                do_wandb,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 1000 == 0:
                logg.info(
                    f"Training at Epoch {epo + 1} iteration {i} with loss {loss.cpu().detach().numpy()}"
                )

        if config.contrastive:
            logg.info(f"Training contrastive at Epoch {epo + 1}")
            for i, batch in tqdm(
                enumerate(contrastive_generator),
                total=len(contrastive_generator),
            ):

                # logg.debug(len(contrastive_generator))
                # logg.debug(batch)
                # logg.debug(batch[0].shape)
                # logg.debug(batch[1].shape)
                # logg.debug(batch[2].shape)
                # return contrastive_datamodule
                # sys.exit(1)

                contrastive_loss_fct = torch.nn.TripletMarginWithDistanceLoss(
                    distance_function=sigmoid_cosine_distance_p,
                    margin=triplet_margin,
                )
                anchor, positive, negative = contrastive_step(
                    model, batch, device
                )
                contrastive_loss = contrastive_loss_fct(
                    anchor, positive, negative
                )

                wandb_log(
                    {
                        "train/c_loss": contrastive_loss,
                        "train/triplet_margin": triplet_margin,
                    },
                    do_wandb,
                )

                opt_contrastive.zero_grad()
                contrastive_loss.backward()
                opt_contrastive.step()

                if i % 1000 == 0:
                    logg.debug(
                        "Training at Epoch "
                        + str(epo + 1)
                        + " iteration "
                        + str(i)
                        + " with loss "
                        + str(contrastive_loss.cpu().detach().numpy())
                    )

            triplet_margin = min(0.5, triplet_margin + (0.5 / config.epochs))
            logg.debug(f"Updating contrastive margin to {triplet_margin}")

        epoch_time_end = time()

        if epo % config.every_n_val == 0:
            with torch.set_grad_enabled(False):

                logg.debug(len(validation_generator))
                val_results = test(
                    model,
                    validation_generator,
                    val_metrics,
                    device,
                    config.classify,
                )

                val_results["epoch"] = epo
                val_results["Charts/epoch_time"] = (
                    epoch_time_end - epoch_time_start
                ) / config.every_n_val

                wandb_log(val_results, do_wandb)

                if val_results[config.watch_metric] > max_metric:
                    model_max = copy.deepcopy(model)
                    max_metric = val_results[config.watch_metric]
                    model_save_path = f"{save_dir}/{config.experiment_id}_best_model_epoch{epo:02}.pt"
                    torch.save(
                        model_max.state_dict(),
                        model_save_path,
                    )
                    logg.info(f"Saving checkpoint model to {model_save_path}")

                logg.info(f"Validation at Epoch {epo + 1}")
                for k, v in val_results.items():
                    if not k.startswith("_"):
                        logg.info(f"{k}: {v}")

    end_time = time()
    logg.info("Beginning testing")
    try:
        with torch.set_grad_enabled(False):
            model_max = model_max.eval()

            test_start_time = time()
            test_results = test(
                model, testing_generator, test_metrics, device, config.classify
            )
            test_end_time = time()

            test_results["epoch"] = epo + 1
            test_results["test/eval_time"] = test_end_time - test_start_time
            test_results["Charts/wall_clock_time"] = end_time - start_time
            wandb_log(test_results, do_wandb)

            logg.info("Final Testing")
            for k, v in test_results.items():
                if not k.startswith("_"):
                    logg.info(f"{k}: {v}")

            model_save_path = (
                f"{save_dir}/{config.experiment_id}_best_model.pt"
            )
            torch.save(
                model_max.state_dict(),
                model_save_path,
            )
            logg.info(f"Saving final model to {model_save_path}")

    except Exception as e:
        logg.error(f"Testing failed with exception {e}")

    return model_max


best_model = main()
