from time import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    auc,
    precision_recall_curve,
)
from torch.autograd import Variable
from tqdm import tqdm
import typing as T

from argparse import ArgumentParser
from src import plm_dti

parser = ArgumentParser(description="PLM_DTI Testing.")
# parser.add_argument('-b', '--batch-size', default=16, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 16), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--exp-id', required=True, help='Experiment ID', dest='experiment_id')
# parser.add_argument('--model-type',required=True, default="SimpleCosine", help='Model architecture', dest='model_type')
parser.add_argument(
    "--mol-feat", required=True, help="Molecule featurizer", dest="mol_feat"
)
parser.add_argument(
    "--prot-feat", required=True, help="Molecule featurizer", dest="prot_feat"
)
# parser.add_argument('--latent-dist', default="Cosine", help='Distance in embedding space to supervise with', dest="latent_dist")
# parser.add_argument('--wandb-proj', required=True, help='Weights and Biases Project', dest='wandb_proj')
# parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
#                     help='number of data loading workers (default: 0)')
# parser.add_argument('--epochs', default=50, type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument(
    "--task",
    choices=["biosnap", "bindingdb", "davis", "biosnap_prot", "biosnap_mol"],
    default="",
    type=str,
    metavar="TASK",
    required=True,
    help="Task name. Could be biosnap, bindingdb, davis, biosnap_prot, biosnap_mol.",
)
# parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
#                     metavar='LR', help='initial learning rate (default: 1e-4)', dest='lr')
# parser.add_argument('--r', '--replicate', default=0, type=int, help='Replicate', dest='replicate')
parser.add_argument(
    "--d", "--device", default=0, type=int, help="CUDA device", dest="device"
)
parser.add_argument(
    "--checkpoint",
    default=None,
    required=True,
    help="Model weights to start from",
)


def flatten(d):
    d_ = {}
    if not isinstance(d, T.Mapping):
        return d
    for k, v in d.items():
        if isinstance(v, T.Mapping):
            d_flat = flatten(v)
            for k_, v_ in d_flat.items():
                d_[k_] = v_
        else:
            d_[k] = v
    return d_


def get_task(task_name):
    if task_name.lower() == "biosnap":
        return "./dataset/BIOSNAP/full_data"
    elif task_name.lower() == "bindingdb":
        return "./dataset/BindingDB"
    elif task_name.lower() == "davis":
        return "./dataset/DAVIS"
    elif task_name.lower() == "biosnap_prot":
        return "./dataset/BIOSNAP/unseen_protein"
    elif task_name.lower() == "biosnap_mol":
        return "./dataset/BIOSNAP/unseen_drug"


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, label) in tqdm(
        enumerate(data_generator), total=len(data_generator)
    ):
        score = model(d.cuda(), p.cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to("cpu").numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision, recall, thresholds = precision_recall_curve(y_label, y_pred)

    all_f1_scores = []
    for t in thresholds:
        all_f1_scores.append(f1_score(y_label, (y_pred >= t).astype(int)))

    thred_optim = thresholds[np.argmax(all_f1_scores)]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = (y_pred >= thred_optim).astype(int)

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print("Confusion Matrix : \n", cm1)
    print("Recall : ", recall_score(y_label, y_pred_s))
    print("Precision : ", precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    # from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print("Accuracy : ", accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print("Sensitivity : ", sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print("Specificity : ", specificity1)

    return (
        roc_auc_score(y_label, y_pred),
        average_precision_score(y_label, y_pred),
        f1_score(y_label, y_pred_s),
        accuracy1,
        sensitivity1,
        specificity1,
        y_pred,
        loss.item(),
    )


def main():
    args = parser.parse_args()
    args.experiment_id = "Testing"
    args.replicate = 0
    config = plm_dti.get_config(
        args.experiment_id, args.mol_feat, args.prot_feat
    )

    device_no = args.device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    torch.cuda.set_device(device)
    print(f"Using CUDA device {device}")

    config.task = args.task
    config.replicate = args.replicate
    torch.manual_seed(args.replicate)  # reproducible torch:2 np:3
    np.random.seed(args.replicate)

    loss_history = []

    print("--- Data Preparation ---")
    config.data.to_disk_path = f"saved_embeddings/{args.task}"
    config.data.device = args.device
    config.data.shuffle = False

    dataFolder = get_task(args.task)

    print("--- loading dataframes ---")
    df_train = pd.read_csv(dataFolder + "/train.csv", header=0, index_col=0)
    df_val = pd.read_csv(dataFolder + "/val.csv", header=0, index_col=0)
    df_test = pd.read_csv(dataFolder + "/test.csv", header=0, index_col=0)

    print("--- loading dataloaders ---")
    (
        training_generator,
        validation_generator,
        testing_generator,
        mol_emb_size,
        prot_emb_size,
    ) = plm_dti.get_dataloaders(df_train, df_val, df_test, **config.data)

    print("--- getting model ---")
    model = torch.load(args.checkpoint)
    model = model.cuda()

    print("--- Go for Testing ---")
    try:
        with torch.set_grad_enabled(False):
            model = model.eval()
            (
                test_auc,
                test_auprc,
                test_f1,
                test_accuracy,
                test_sensitivity,
                test_specificity,
                test_logits,
                test_loss,
            ) = test(testing_generator, model)
            print(
                "Testing AUROC: "
                + str(test_auc)
                + " , AUPRC: "
                + str(test_auprc)
                + " , F1: "
                + str(test_f1)
                + " , Test loss: "
                + str(test_loss)
            )
    except Exception as e:
        print(f"testing failed with exception {e}")
    return model, loss_history


s = time()
model, loss_history = main()
e = time()
print(e - s)
