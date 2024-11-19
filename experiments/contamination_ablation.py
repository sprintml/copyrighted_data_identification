import os
import sys

sys.path.append(".")
sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from torch import Tensor as T
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from itertools import product

from src import get_p_value
from src.attacks import get_datasets_clf

from experiments.utils import (
    set_plt,
    ATTACKS_NAME_MAPPING,
    MODELS_NAME_MAPPING,
    MODELS_ORDER,
    RESAMPLING_CNT,
    MODELS,
    RUN_ID,
    OURS,
)

set_plt()

ATTACKS = ["cdi"]
NOISE_RATIOS = [0.0, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
NSAMPLES_PLOT = [30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]
MODELS_TO_PLOT = ["LDM256", "DiT512", "U-ViT256-T2I"]
PATH_TO_SCORES = "out/scores"
PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "experiments/out/noise_influence"
features_indices = np.arange(26)

nsamples = np.array(NSAMPLES_PLOT)


def get_data_attack(
    members: np.ndarray,
    nonmembers: np.ndarray,
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

    assert (
        len(members_scores)
        == len(nonmembers_scores)
        == (len(test_dataset.data) + len(train_dataset.data)) // 2
    )
    return members_scores, nonmembers_scores


def plot_noise_pvalue(model: str, df: pd.DataFrame, ax: plt.Axes, idx: int) -> None:
    sns.lineplot(
        data=df,
        x="noise_ratio",
        y="pvalue",
        hue="n",
        ax=ax,
        legend="full",
        palette="cool",
        hue_norm=matplotlib.colors.LogNorm(),
    )
    ax.set(
        title=model,
        yscale="log",
        ylim=[1e-5, 0.55],
    )
    ax.plot(
        df.noise_ratio,
        0.05 * np.ones_like(df.noise_ratio),
        "--",
        color="black",
        label="p-value: 0.05",
    )
    ax.plot(
        df.noise_ratio,
        0.01 * np.ones_like(df.noise_ratio),
        "--",
        color="green",
        label="p-value: 0.01",
    )
    if idx:
        ax.set(
            xlabel="Contamination ratio",
            ylabel="",
        )
    else:
        ax.set(
            xlabel="Contamination ratio",
            ylabel="p-value",
        )
    ax.legend().remove()


def get_data() -> list:
    out = []
    for attack, model, noise_ratio in tqdm(product(ATTACKS, MODELS, NOISE_RATIOS)):
        data = np.load(
            f"{PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}.npz", allow_pickle=True
        )
        members_lower = data["metadata"][()]["members_lower"]
        n_eval_samples = data["metadata"][()]["n_samples_eval"]
        members = torch.from_numpy(data["members"][:, 0, features_indices])
        nonmembers = torch.from_numpy(data["nonmembers"][:, 0, features_indices])

        nsamples_run = nsamples[nsamples <= n_eval_samples]
        if noise_ratio:
            nsamples_run = nsamples_run[
                ((1 + noise_ratio) * nsamples_run).astype(int)
                == (1 + noise_ratio) * nsamples_run
            ]  # to ensure that we have precise ratio of "noise", e.g., 0.01 and 2 samples will not work out

        indices_total = np.arange(len(members))
        for n in nsamples_run:
            for _ in range(RESAMPLING_CNT):
                noise_samples = int(n * noise_ratio)
                indices = np.random.choice(
                    indices_total, n + noise_samples, replace=False
                )

                if noise_samples:
                    sample_members = np.concatenate(
                        [
                            members[indices[: n - noise_samples]],
                            nonmembers[indices[n : n + noise_samples]],
                        ]
                    )
                    sample_nonmembers = nonmembers[indices[:n]]
                    sample_members = torch.from_numpy(sample_members)
                else:
                    sample_members = members[indices[:n]]
                    sample_nonmembers = nonmembers[indices[:n]]

                assert len(sample_members) == len(sample_nonmembers) == n
                members_scores, nonmembers_scores = get_data_attack(
                    sample_members, sample_nonmembers
                )
                pvalue, is_correct_order = get_p_value(
                    members_scores, nonmembers_scores, members_lower=members_lower
                )
                out.append(
                    [
                        ATTACKS_NAME_MAPPING[attack],
                        MODELS_NAME_MAPPING[model],
                        n,
                        noise_ratio,
                        pvalue,
                        is_correct_order,
                    ]
                )

    return out


def get_noise_pvalue_cmp(df: pd.DataFrame):
    fig, axs = plt.subplots(
        1, len(MODELS_TO_PLOT), figsize=(10 * len(MODELS_TO_PLOT), 5 * 1)
    )
    axs = axs.flatten()

    tmp_df = (
        df.loc[(df.Attack == OURS) & (df.n.isin(NSAMPLES_PLOT))]
        .groupby(["Model", "noise_ratio", "n"])
        .pvalue.mean()
        .reset_index()
    )
    for idx, (model, ax) in tqdm(enumerate(zip(MODELS_TO_PLOT, axs))):
        plot_noise_pvalue(model, tmp_df[tmp_df.Model == model], ax, idx)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        ncols=1,
        title="Number of samples in $\mathbf{Q}_{sus}$",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=18,
        title_fontsize=18,
    )

    plt.savefig(
        f"{PATH_TO_PLOTS}/noise_vs_pvalues.pdf", format="pdf", bbox_inches="tight"
    )

def get_fp_table(df: pd.DataFrame):
    df = (
        df.loc[(df.Attack == OURS) & (df.n == 10_000) & (df.noise_ratio.isin([0, 1]))]
        .groupby(["noise_ratio", "Model"])
        .pvalue.mean()
        .reset_index()
    )
    from math import log10, floor

    def find_exp(number) -> int:
        base10 = log10(abs(number))
        return floor(base10)

    df = df.pivot(index="noise_ratio", columns="Model", values="pvalue").applymap(
        lambda x: f"$10^{{{find_exp(x)}}}$" if (x < 0.1 and x != 0) else f"{x:.2f}"
    )[MODELS_ORDER]
    df.index.name = ""
    df.columns.name = ""
    df.to_latex(f"{PATH_TO_PLOTS}/fp_table.tex", escape=False)


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    np.random.seed(42)
    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv")
    except FileNotFoundError:
        data = get_data()
        df = pd.DataFrame(
            data,
            columns=[
                "Attack",
                "Model",
                "n",
                "noise_ratio",
                "pvalue",
                "is_correct_order",
            ],
        )
        df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)

    get_fp_table(df)
    get_noise_pvalue_cmp(df)


if __name__ == "__main__":
    main()
