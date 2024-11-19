import os
import sys

sys.path.append(".")
sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from typing import Tuple
from torch import Tensor as T

import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product

from src import get_p_value
from src.attacks import get_datasets_clf 
from experiments.utils import (
    set_plt,
    ATTACKS_NAME_MAPPING,
    MODELS_NAME_MAPPING,
    MODELS_COLORS,
    MODELS_ORDER,
    RESAMPLING_CNT,
    MODELS,
    RUN_ID,
    OURS,
)

set_plt()

ATTACKS = ["cdi"]
PVALUES = [0.01, 0.05]
NSAMPLES_TO_SHOW = [100, 500, 1000, 5000]
features_indices = np.arange(26)

PATH_TO_SCORES = "out/scores"
PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "experiments/out/pvalue_per_sample"


def get_data_attack(
    members: np.ndarray, nonmembers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    members_indices = np.arange(len(members))    
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    members_scores, nonmembers_scores = [], []

    for train_idx, test_idx in kf.split(members_indices):
        # Get train and test splits for members and nonmembers using the same indices
        members_train = members[train_idx]
        members_test = members[test_idx]
        nonmembers_train = nonmembers[train_idx]
        nonmembers_test = nonmembers[test_idx]
        
        # Create training and testing datasets using get_datasets_clf
        train_dataset = get_datasets_clf(members_train, nonmembers_train)
        test_dataset = get_datasets_clf(members_test, nonmembers_test)
        
        # Standardize the data
        ss = StandardScaler()
        train_data = torch.from_numpy(ss.fit_transform(train_dataset.data))
        test_data = torch.from_numpy(ss.transform(test_dataset.data))
        
        # Train the classifier
        clf = LogisticRegression(random_state=0, max_iter=1000, n_jobs=None, solver="liblinear")
        clf.fit(train_data, train_dataset.label)
        
        # Compute scores only for the test dataset
        scores = torch.from_numpy(clf.predict_proba(test_data)[:, 1])
        
        # Separate the scores for members and nonmembers
        members_scores.append(scores[: len(test_dataset.data) // 2])
        nonmembers_scores.append(scores[len(test_dataset.data) // 2 :])
        
    members_scores = torch.cat(members_scores)
    nonmembers_scores = torch.cat(nonmembers_scores)

    assert len(members_scores) == len(nonmembers_scores) == (len(test_dataset.data) + len(train_dataset.data)) // 2
    return members_scores, nonmembers_scores
    



def plot_pvalues_cmp(
    title: str,
    df: pd.DataFrame,
    ax: plt.Axes,
    hue: str,
    xlabel: str = "Number of samples",
):
    sns.lineplot(
        data=df,
        x="n",
        y="pvalue",
        hue=hue,
        # style=hue,
        ax=ax,
        palette=(
            [MODELS_COLORS[model] for model in MODELS_ORDER] if hue == "Model" else None
        ),
        # markers=MODELS_MARKERS,
    )
    ax.plot(df.n, 0.05 * np.ones_like(df.n), "--", color="black", label="p-value: 0.05")
    ax.plot(df.n, 0.01 * np.ones_like(df.n), "--", color="green", label="p-value: 0.01")
    ax.set(
        xscale="log",
        yscale="log",
        ylim=[10 ** (-3), 1],
        title=title,
        ylabel="p-value",
        xlabel=xlabel,
    )
    ax.get_legend().remove()


def get_data() -> list:
    out = []
    for attack, model in tqdm(product(ATTACKS, MODELS)):
        data = np.load(
            f"{PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}.npz", allow_pickle=True
        )
        members_lower = data["metadata"][()]["members_lower"]
        n_eval_samples = data["metadata"][()]["n_samples_eval"]

        members = torch.from_numpy(data["members"][:, 0, features_indices])
        nonmembers = torch.from_numpy(data["nonmembers"][:, 0, features_indices])

        nsamples = np.array(
            [n for n in range(5, 11)]
            + [n for n in range(20, 110, 10)]
            + [n for n in range(200, 1100, 100)]
            + [n for n in range(2000, 11000, 1000)]
            + [20000]
        )
        nsamples = nsamples[nsamples <= n_eval_samples]
        indices_total = np.arange(n_eval_samples)
        for n in nsamples:
            for r in range(RESAMPLING_CNT):
                indices = np.random.permutation(indices_total)
                members_scores, nonmembers_scores = get_data_attack(members[indices[:n]], nonmembers[indices[:n]])
                pvalue, is_correct_order = get_p_value(
                    members_scores, nonmembers_scores, members_lower=members_lower
                )
                out.append(
                    [
                        ATTACKS_NAME_MAPPING[attack],
                        MODELS_NAME_MAPPING[model],
                        n,
                        pvalue,
                        is_correct_order,
                        r,
                    ]
                )

    return out


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    np.random.seed(42)
    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv")
    except FileNotFoundError:
        data = get_data()
        df = pd.DataFrame(
            data, columns=["Attack", "Model", "n", "pvalue", "is_correct_order", "r"]
        )
        df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10 * 1, 3 * 1),
    )

    plot_pvalues_cmp(
        None,
        df.loc[df.Attack == OURS],
        ax,
        hue="Model",
        xlabel="Number of samples in $\mathbf{Q}_{sus}$",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        ncols=1,
        title="Model",
        bbox_to_anchor=(1.15, 0.5),
    )
    plt.savefig(
        f"{PATH_TO_PLOTS}/pvalue_per_sample_{OURS}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

if __name__ == "__main__":
    main()
