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

from argparse import ArgumentParser

import wandb
from omegaconf import OmegaConf

from src import featurizers
from src import architectures
from src.utils import set_random_seed, config_logger
from src.data import get_task_dir, DTIDataModule

parser = ArgumentParser(description='PLM_DTI Training.')
parser.add_argument('--exp-id', required=True, help='Experiment ID', dest='experiment_id')
parser.add_argument('--config', required=True, help='YAML config file')
parser.add_argument('--wandb-proj', required=True, help='Weights and Biases Project', dest='wandb_proj')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'davis', 'biosnap_prot', 'biosnap_mol'],
                    default='', type=str, required=True,
                    help='Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol.'
                   )

# parser.add_argument('--model-type',required=True, default="SimpleCosine", 
#                     help='Model architecture', dest='model_type'
#                    )
parser.add_argument('--drug-feat', 
                    help='Drug featurizer', dest='drug_featurizer'
                   )
parser.add_argument('--target-feat', 
                    help='Target featurizer', dest='target_featurizer'
                   )
parser.add_argument('--distance-metric',
                    help='Distance in embedding space to supervise with', dest="distance_metric"
                   )
# parser.add_argument('-j', '--workers', default=0, type=int, 
#                     help='number of data loading workers (default: 0)'
#                    )
parser.add_argument('--epochs', type=int, 
                    help='number of total epochs to run'
                   )
parser.add_argument('-b', '--batch-size', type=int, 
                    help='batch size'
                   )
parser.add_argument('--lr', '--learning-rate', type=float,
                    help='initial learning rate (default: 1e-4)', dest='lr'
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

def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, label) in tqdm(enumerate(data_generator),total=len(data_generator)):
        score = model(d.cuda(), p.cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision, recall, thresholds = precision_recall_curve(y_label, y_pred)

    all_f1_scores = []
    for t in thresholds:
        all_f1_scores.append(f1_score(y_label,(y_pred >= t).astype(int)))

    thred_optim = thresholds[np.argmax(all_f1_scores)]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = (y_pred >= thred_optim).astype(int)

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                              y_pred_s), accuracy1, sensitivity1, specificity1, y_pred, loss.item()


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
    
    datamodule = DTIDataModule(
        task_dir,
        drug_featurizer,
        target_featurizer,
        device = device
    )
    datamodule.setup()

    # Load DataLoaders
    logg.info("Getting DataLoaders")
    training_generator = datamodule.train_dataloader()
    validation_generator = datamodule.val_dataloader()
    testing_generator = datamodule.test_dataloader()

    config.drug_shape = drug_featurizer.shape
    config.target_shape = target_featurizer.shape
    
    logg.info("Creating model")
    if "checkpoint" not in config:
        model = getattr(architectures, "SimpleCoembedding")(
            config.drug_shape,
            config.target_shape,
            latent_size = config.latent_dimension,
            latent_distance = config.latent_distance
        )
    else:
        model = torch.load(config.checkpoint)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    print('--- loading wandb ---')
    wandb.init(
        project=config.wandb_proj,
        name=config.experiment_id,
        config=config,
    )
    wandb.watch(model, log_freq=100)

    # early stopping
    max_auc = 0
    max_auprc = 0
    loss_history = []
    model_max = copy.deepcopy(model)

    # with torch.set_grad_enabled(False):
    #     auc, auprc, f1, logits, loss = test(testing_generator, model_max)
    #     # wandb.log({"test/loss": loss, "epoch": 0,
    #     #            "test/auc": float(auc),
    #     #            "test/aupr": float(auprc),
    #     #            "test/f1": float(f1),
    #     #           })
    #     print('Initial Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(
    #         f1) + ' , Test loss: ' + str(loss))

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    tg_len = len(training_generator)
    start_time = time()
    for epo in range(config.epochs):
        model.train()
        epoch_time_start = time()
        for i, (d, p, label) in enumerate(training_generator):
            score = model(d.cuda(), p.cuda())

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss = loss_fct(n, label)
            loss_history.append(loss)
            wandb.log({"train/loss": loss, "epoch": epo,
                       "step": epo*tg_len*config.batch_size + i*config.batch_size
                  })

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        epoch_time_end = time()
        if epo % config.every_n_val == 0:
            with torch.set_grad_enabled(False):
                val_auc, val_auprc, val_f1, val_accuracy, val_sensitivity, val_specificity, val_logits, val_loss = test(validation_generator, model)
                wandb.log({"val/loss": val_loss, "epoch": epo,
                           "val/auc": float(val_auc),
                           "val/aupr": float(val_auprc),
                           "val/f1": float(val_f1),
                           "val/acc": float(val_accuracy),
                           "val/sens": float(val_sensitivity),
                           "val/spec": float(val_specificity),
                           "Charts/epoch_time": (epoch_time_end - epoch_time_start)/config.every_n_val
                  })
                if val_auprc > max_auprc:
                    model_max = copy.deepcopy(model)
                    max_auprc = val_auprc
                print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(val_auc) + ' , AUPRC: ' + str(
                    val_auprc) + ' , F1: ' + str(val_f1))

    end_time = time()
    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            model_max = model_max.eval()
            test_start_time = time()
            test_auc, test_auprc, test_f1, test_accuracy, test_sensitivity, test_specificity, test_logits, test_loss = test(testing_generator, model_max)
            test_end_time = time()
            wandb.log({"test/loss": test_loss, "epoch": epo,
                       "test/auc": float(test_auc),
                       "test/aupr": float(test_auprc),
                       "test/f1": float(test_f1),
                       "test/acc": float(test_accuracy),
                       "test/sens": float(test_sensitivity),
                       "test/spec": float(test_specificity),
                       "test/eval_time": (test_end_time - test_start_time),
                       "Charts/wall_clock_time": (end_time - start_time)
            })
            print(
                'Testing AUROC: ' + str(test_auc) + ' , AUPRC: ' + str(test_auprc) + ' , F1: ' + str(test_f1) + ' , Test loss: ' + str(
                    test_loss))
            # trained_model_artifact = wandb.Artifact(conf.experiment_id, type="model")
            torch.save(model_max, f"best_models/{config.experiment_id}_best_model.sav")
    except:
        print('testing failed')
    return model_max, loss_history


s = time()
config = main()
e = time()
print(e - s)
