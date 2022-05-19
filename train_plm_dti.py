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
import plm_dti
import wandb

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='PLM_DTI Training.')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--exp-id', required=True, help='Experiment ID', dest='experiment_id')
parser.add_argument('--model-type',required=True, default="SimpleCosine", help='Model architecture', dest='model_type')
parser.add_argument('--mol-feat', required=True, help='Molecule featurizer', dest='mol_feat')
parser.add_argument('--prot-feat', required=True, help='Molecule featurizer', dest='prot_feat')
parser.add_argument('--latent-dist', default="Cosine", help='Distance in embedding space to supervise with', dest="latent_dist")
parser.add_argument('--wandb-proj', required=True, help='Weights and Biases Project', dest='wandb_proj')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'davis', 'biosnap_prot', 'biosnap_mol'],
                    default='', type=str, metavar='TASK', required=True,
                    help='Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol.')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-4)', dest='lr')
parser.add_argument('--r', '--replicate', default=0, type=int, help='Replicate', dest='replicate')
parser.add_argument('--d', '--device', default=0, type=int, help='CUDA device', dest='device')

def flatten(d):
    d_ = {}
    if not isinstance(d, T.Mapping):
        return d
    for k,v in d.items():
        if isinstance(v, T.Mapping):
            d_flat = flatten(v)
            for k_,v_ in d_flat.items():
                d_[k_] = v_
        else:
            d_[k] = v
    return d_

def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'
    elif task_name.lower() == 'biosnap_prot':
        return './dataset/BIOSNAP/unseen_protein'
    elif task_name.lower() == 'biosnap_mol':
        return './dataset/BIOSNAP/unseen_drug'


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
        all_f1_scores.append(f1_score(y_label,(y_pred >= t).astype(int), t))

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
    args = parser.parse_args()
    config = plm_dti.get_config(args.experiment_id, args.mol_feat, args.prot_feat)

    print(f"Using CUDA device {device}")

    config.task = args.task
    config.replicate = args.replicate
    torch.manual_seed(args.replicate)  # reproducible torch:2 np:3
    np.random.seed(args.replicate)
    config.model.model_type = args.model_type
    if config.model.model_type == "LSTMCosine":
        config.data.pool = False
    else:
        config.data.pool = True
    config.training.lr = args.lr
    config.training.n_epochs = args.epochs
    config.data.batch_size = args.batch_size

    loss_history = []

    print('--- Data Preparation ---')
    config.data.batch_size = args.batch_size
    config.data.shuffle = True
    config.data.num_workers = args.workers
    config.data.drop_last = True
    config.data.to_disk_path = f"saved_embeddings/{args.task}"
    config.data.device = args.device

    dataFolder = get_task(args.task)

    print('--- loading dataframes ---')
    df_train = pd.read_csv(dataFolder + '/train.csv',header=0,index_col=0)
    df_val = pd.read_csv(dataFolder + '/val.csv',header=0,index_col=0)
    df_test = pd.read_csv(dataFolder + '/test.csv',header=0,index_col=0)

    print('--- loading dataloaders ---')
    training_generator, validation_generator, testing_generator, mol_emb_size, prot_emb_size = plm_dti.get_dataloaders(
        df_train,
        df_val,
        df_test,
        **config.data
    )

    config.model.mol_emb_size, config.model.prot_emb_size = mol_emb_size, prot_emb_size
    config.model.distance_metric = args.latent_dist
    model = plm_dti.get_model(**config.model)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    wandb.init(
        project=args.wandb_proj,
        name=config.experiment_id,
        config=flatten(config),
    )
    wandb.watch(model, log_freq=100)

    # early stopping
    max_auc = 0
    max_auprc = 0
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
    for epo in range(config.training.n_epochs):
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
                       "step": epo*tg_len*args.batch_size + i*args.batch_size
                  })

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        epoch_time_end = time()
        if epo % config.training.every_n_val == 0:
            with torch.set_grad_enabled(False):
                val_auc, val_auprc, val_f1, val_accuracy, val_sensitivity, val_specificity, val_logits, val_loss = test(validation_generator, model)
                wandb.log({"val/loss": val_loss, "epoch": epo,
                           "val/auc": float(val_auc),
                           "val/aupr": float(val_auprc),
                           "val/f1": float(val_f1),
                           "val/acc": float(val_accuracy),
                           "val/sens": float(val_sensitivity),
                           "val/spec": float(val_specificity),
                           "Charts/epoch_time": (epoch_time_end - epoch_time_start)/config.training.every_n_val
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
model_max, loss_history = main()
e = time()
print(e - s)
