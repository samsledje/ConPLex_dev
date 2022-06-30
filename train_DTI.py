import copy
from time import time
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc, precision_recall_curve
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import typing as T
import torchmetrics

from argparse import ArgumentParser

import wandb
from omegaconf import OmegaConf

from src import featurizers
from src import architectures
from src.utils import set_random_seed, config_logger, sigmoid_cosine_distance_p
from src.data import get_task_dir, DTIDataModule, DUDEDataModule

parser = ArgumentParser(description='PLM_DTI Training.')
parser.add_argument('--exp-id', required=True, help='Experiment ID', dest='experiment_id')
parser.add_argument('--config', required=True, help='YAML config file')
parser.add_argument('--wandb-proj', required=True, help='Weights and Biases Project', dest='wandb_proj')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'davis', 'biosnap_prot', 'biosnap_mol'],
                    type=str,
                    help='Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol.'
                   )
parser.add_argument('--no-contrastive', action='store_true')

parser.add_argument('--drug-featurizer',
                    help='Drug featurizer', dest='drug_featurizer'
                   )
parser.add_argument('--target-featurizer',
                    help='Target featurizer', dest='target_featurizer'
                   )
parser.add_argument('--distance-metric',
                    help='Distance in embedding space to supervise with', dest="distance_metric"
                   )
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run'
                   )
parser.add_argument('-b', '--batch-size', type=int,
                    help='batch size'
                   )
parser.add_argument('--lr', '--learning-rate', type=float,
                    help='initial learning rate', dest='lr'
                   )
parser.add_argument('--clr', type=float,
                    help='initial learning rate', dest='clr'
                   )
parser.add_argument('--r', '--replicate', type=int,
                    help='Replicate', dest='replicate'
                   )
parser.add_argument('--d', '--device', type=int,
                    help='CUDA device', dest='device'
                   )
parser.add_argument('--verbosity', type=int,
                    help='Level to log at', dest='verbosity'
                   )
parser.add_argument('--checkpoint', default=None,
                    help='Model weights to start from'
                   )

def test(model, data_generator, metrics, device=None):

    if device is None:
        device = torch.device("cpu")

    for k, metric in metrics.items():
        metric.reset()
        metrics[k] = metric.to(device)

    model.eval()

    for i, batch in tqdm(enumerate(data_generator),total=len(data_generator)):

        loss_fct = torch.nn.BCELoss()
        pred, label = step(model, batch, device)
        loss = loss_fct(pred, label)

        for _, metric in metrics.items():
            metric(pred, label.int())
    
    results = {k: metric.compute() for (k, metric) in metrics.items()}
    for metric in metrics.values(): 
        metrics[k] = metric.to("cpu")

    return results

def step(model, batch, device = None):

    if device is None:
        device = torch.device("cpu")

    drug, target, label = batch

    sigmoid = torch.nn.Sigmoid()
    score = model(drug.to(device), target.to(device))
    label = Variable(torch.from_numpy(np.array(label)).float()).to(device)
    pred = torch.squeeze(sigmoid(score))
    return pred, label

def contrastive_step(model, batch, device = None):
     
        if device is None:
            device = torch.device("cpu")
            
        anchor, positive, negative = batch
        
        anchor_projection = model.target_projector(anchor.to(device))
        positive_projection = model.drug_projector(positive.to(device))
        negative_projection = model.drug_projector(negative.to(device))
        
        return anchor_projection, positive_projection, negative_projection

def main():
    # Get configuration
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    arg_overrides = {k:v for k,v in vars(args).items() if v is not None}
    config.update(arg_overrides)

    # Logging
    logg = config_logger(
        None,
        "%(asctime)s [%(levelname)s] %(message)s",
        config.verbosity,
        use_stdout=True,
    )
    logg.propagate = False

    # Set CUDA device
    device_no = config.device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    logg.info(f"Using CUDA device {device}")

    # Set random state
    logg.debug("Setting random state")
    set_random_seed(config.replicate)

    # Load DataModule
    logg.info("Preparing DataModule")
    task_dir = get_task_dir(config.task)

    drug_featurizer = getattr(featurizers, config.drug_featurizer)(save_dir = task_dir)
    target_featurizer = getattr(featurizers, config.target_featurizer)(save_dir = task_dir)

    if task == "dti_dg":
        datamodule = 
    else:
        datamodule = DTIDataModule(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device = device
        )
    datamodule.prepare_data()
    datamodule.setup()

    # Load DataLoaders
    logg.info("Getting DataLoaders")
    training_generator = datamodule.train_dataloader()
    validation_generator = datamodule.val_dataloader()
    testing_generator = datamodule.test_dataloader()
    
    if not config.no_contrastive:
        logg.info("Loading contrastive data (DUDE)")
        dude_drug_featurizer = getattr(featurizers, config.drug_featurizer)(save_dir = './dataset/DUDe/')
        dude_target_featurizer = getattr(featurizers, config.target_featurizer)(save_dir = './dataset/DUDe/')
        
        contrastive_datamodule = DUDEDataModule(
            config.contrastive_split,
            dude_drug_featurizer,
            dude_target_featurizer,
            device = device
        )
        contrastive_datamodule.setup(stage="fit")
        contrastive_generator = contrastive_datamodule.train_dataloader()

    config.drug_shape = drug_featurizer.shape
    config.target_shape = target_featurizer.shape

    # Create model
    logg.info("Creating model")
    if "checkpoint" not in config:
        model = getattr(architectures, "SimpleCoembedding")(
            config.drug_shape,
            config.target_shape,
            latent_dimension = config.latent_dimension,
            latent_distance = config.latent_distance
        )
    else:
        model = torch.load(config.checkpoint)

    model = model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr = config.lr)
    if not config.no_contrastive:
        opt_contrastive = torch.optim.Adam(model.parameters(), lr = config.clr)
    
    # Initialize wandb
    logg.debug(f"Initializing wandb project {config.wandb_proj}")
    wandb.init(
        project=config.wandb_proj,
        name=config.experiment_id,
        config=dict(config),
    )
    wandb.watch(model, log_freq=100)

    # Metrics
    logg.debug(f"Creating metrics")
    max_aupr = 0
    triplet_margin = 0
    model_max = copy.deepcopy(model)
    model_save_dir = config.get("model_save_dir", ".")

    val_metrics = {
        "val/AUPR": torchmetrics.AveragePrecision(),
        "val/AUROC": torchmetrics.AUROC(),
        "val/F1": torchmetrics.F1Score(),
    }

    test_metrics = {
        "test/AUPR": torchmetrics.AveragePrecision(),
        "test/AUROC": torchmetrics.AUROC(),
        "test/F1": torchmetrics.F1Score(),
    }

    logg.info("Beginning Training")

    torch.backends.cudnn.benchmark = True
    tg_len = len(training_generator)

    start_time = time()
    for epo in range(config.epochs):
        model.train()
        epoch_time_start = time()

        for i, batch in enumerate(training_generator):

            loss_fct = torch.nn.BCELoss()
            pred, label = step(model, batch, device)
            loss = loss_fct(pred, label)


            wandb.log({"epoch": epo+1,
                       "train/loss": loss,
                       "step": (epo * tg_len * config.batch_size) + (i * config.batch_size),
                  })

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i % 1000 == 0):
                logg.info(f"Training at Epoch {epo + 1} iteration {i} with loss {loss.cpu().detach().numpy()}")

        epoch_time_end = time()
        
        if not config.no_contrastive:
            for i, batch in enumerate(contrastive_generator):
                
                contrastive_loss_fct = torch.nn.TripletMarginWithDistanceLoss(
                    distance_function=sigmoid_cosine_distance_p,
                    margin=triplet_margin,
                )
                anchor, positive, negative = contrastive_step(model, batch, device)
                contrastive_loss = contrastive_loss_fct(anchor, positive, negative)
                
                wandb.log({"epoch": epo+1,
                           "train/c_loss": contrastive_loss
                          })
                
                opt_contrastive.zero_grad()
                contrastive_loss.backward()
                opt_contrastive.step()

            triplet_margin = min(0.5, triplet_margin + (0.5 / config.n_epochs))
            logg.debug(f"Updating contrastive margin to {triplet_margin}")
                                                
        if epo % config.every_n_val == 0:
            with torch.set_grad_enabled(False):

                val_results = test(model, validation_generator, val_metrics, device)
                val_results["epoch"] = epo
                val_results["Charts/epoch_time"]  = (epoch_time_end - epoch_time_start) / config.every_n_val

                wandb.log(val_results)

                if val_results["val/AUPR"] > max_aupr:
                    model_max = copy.deepcopy(model)
                    max_aupr = val_results["val/AUPR"]
                    torch.save(model_max, f"{model_save_dir}/{config.experiment_id}_best_model_epoch{epo}.sav")

                logg.info(f"Validation at Epoch {epo + 1} AUROC: {val_results['val/AUROC']}, AUPR: {val_results['val/AUPR']}, F1: {val_results['val/F1']}")

    end_time = time()
    logg.info("Beginning testing")
    try:
        with torch.set_grad_enabled(False):
            model_max = model_max.eval()

            test_start_time = time()
            test_results = test(model, testing_generator, test_metrics, device)
            test_end_time = time()

            test_results["test/eval_time"] = (test_end_time - test_start_time)
            test_results["Charts/wall_clock_time"] = (end_time - start_time)
            wandb.log(test_results)

            logg.info(f"Test AUROC: {test_results['test/AUROC']}, AUPR: {test_results['test/AUPR']}, F1: {test_results['test/F1']}")
            torch.save(model_max, f"{model_save_dir}/{config.experiment_id}_best_model.sav")

    except Exception as e:
        logg.error(f"Testing failed with exception {e}")

    return model_max

s = time()
best_model = main()
e = time()
print(e - s)
