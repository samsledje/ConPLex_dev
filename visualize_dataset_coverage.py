import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from src.data import get_task_dir


def calc_coverage(
    df, drug_col="SMILES", target_col="Target Sequence", label_col="Label"
):
    drug_uniq = list(df[drug_col].unique())
    target_uniq = list(df[target_col].unique())

    coverage_mtx = np.zeros((len(drug_uniq), len(target_uniq)))

    # for i, d in tqdm(enumerate(drug_uniq),leave=False, total=len(drug_uniq)):
    #     for j, t in tqdm(enumerate(target_uniq),leave=False, total=len(target_uniq)):
    #         obs_df = df[(df[drug_col] == d) & (df[target_col] == t)]
    #         n_obs = len(obs_df)
    #         coverage_mtx[i,j] = n_obs

    for _, r in tqdm(df.iterrows(), leave=False, total=len(df)):
        i = drug_uniq.index(r[drug_col])
        j = target_uniq.index(r[target_col])
        coverage_mtx[i, j] = coverage_mtx[i, j] + 1

    return coverage_mtx


def viz_coverage(
    df,
    drug_col="SMILES",
    target_col="Target Sequence",
    label_col="Label",
    precomputed=None,
    key="",
):
    if precomputed is None:
        coverage_mtx = calc_coverage(df, drug_col, target_col, label_col)
    else:
        coverage_mtx = precomputed

    cmap = "binary"
    vmin = 0
    vmax = 1
    drug_uniq = df[drug_col].unique()
    target_uniq = df[target_col].unique()
    df = pd.DataFrame(coverage_mtx, index=drug_uniq, columns=target_uniq)
    df.to_csv(f"img/{key}_coverage_mtx.tsv", sep="\t")

    sns.set(rc={"figure.figsize": (11.7, 8.27)}, style="whitegrid")
    sns.heatmap(
        df,
        xticklabels=False,
        yticklabels=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.ylabel("Drug")
    plt.xlabel("Target")
    plt.title(f"{key} Drug x Target Coverage")
    plt.savefig(f"img/{key}_coverage_drugxtarget.png", bbox_inches="tight")
    # plt.show()
    plt.close()

    drug_cov = coverage_mtx.mean(axis=1)
    target_cov = coverage_mtx.mean(axis=0)

    coverages = np.concatenate([drug_cov, target_cov])
    labels = ["Drug"] * len(drug_cov) + ["Target"] * len(target_cov)
    covg_df = pd.DataFrame({"Coverage": coverages, "Type": labels})
    # sns.displot(covg_df, x="Coverage", hue="Type")
    sns.boxplot(x=covg_df["Coverage"], y=covg_df["Type"])
    plt.title(f"{key} Distribution of Coverages")
    plt.xlim(0, 1)
    plt.savefig(f"img/{key}_coverage_distribution.png", bbox_inches="tight")
    # plt.show()
    plt.close()


if __name__ == "__main__":

    dti_tasks = [
        "davis",
        "biosnap",
        "biosnap_prot",
        "biosnap_mol",
        "bindingdb",
    ]

    enzpred_tasks = {
        "halogenase": "./dataset/EnzPred/halogenase_NaBr_binary.csv",
        "gt": "./dataset/EnzPred/gt_acceptors_achiral_binary.csv",
        "bkace": "./dataset/EnzPred/duf_binary.csv",
        "esterase": "./dataset/EnzPred/esterase_binary.csv",
        "phosphatase": "./dataset/EnzPred/phosphatase_chiral_binary.csv",
        "kinase": "./dataset/EnzPred/davis_filtered.csv",
    }

    for k, v in enzpred_tasks.items():
        print(f"======== {k} =======")
        df = pd.read_csv(v, index_col=0)
        drug_col = df.columns[1]
        target_col = df.columns[0]
        label_col = df.columns[2]
        viz_coverage(
            df,
            drug_col,
            target_col,
            label_col,
            key=k,
        )

    for k in dti_tasks:
        print(f"======== {k} =======")
        task_dir = get_task_dir(k)
        df = pd.read_csv(task_dir / "train.csv")
        drug_col = df.columns[1]
        target_col = df.columns[0]
        label_col = df.columns[2]
        viz_coverage(
            df,
            drug_col,
            target_col,
            label_col,
            key=k,
        )
