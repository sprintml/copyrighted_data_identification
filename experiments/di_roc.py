import pandas as pd
import matplotlib.pyplot as plt

from utils import set_plt, OURS, MODELS_COLORS
from sklearn.metrics import roc_curve
import numpy as np
import os

from itertools import product
from tqdm import tqdm


set_plt()

PATH_TO_PLOTS = "experiments/plots/di_roc/"
MODELS = ["DiT256", "U-ViT256-T2I"]
nsamples = [30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]
attacks = ["Denoising Loss", "SecMI$_{stat}$", "PIA", "PIAN", OURS]


def get_di_only_curve(positive: pd.DataFrame, negative: pd.DataFrame):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for ax, model in zip(axs, positive.Model.unique()):
        tprs_at_fpr_1 = []
        for n in nsamples:
            tmp_positive = positive.loc[(positive.Model == model) & (positive.n == n)]
            tmp_negative = negative.loc[(negative.Model == model) & (negative.n == n)]
            fpr, tpr, _ = roc_curve(
                [1] * tmp_positive.shape[0] + [0] * tmp_negative.shape[0],
                list(1 - tmp_positive.pvalue) + list(1 - tmp_negative.pvalue),
            )
            ax.plot(fpr, tpr, label=n)
            tprs_at_fpr_1.append(f"{n}: {tpr[np.sum(fpr < 0.01)]}")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim([1e-3, 1.1])
        ax.set_xlim([1e-3, 1.1])
        ax.set_title(model)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            tprs_at_fpr_1 + ["Random: 0.01"],
            loc="lower right",
            title="TPR@FPR=1%",
        )

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles[:-1],
        labels[:-1],
        loc="lower center",
        ncol=len(negative.n.unique()) + 1,
        title="$\mathbf{Q_{sus}}$ size",
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.tight_layout()
    plt.savefig(f"{PATH_TO_PLOTS}di_roc_curve.pdf", bbox_inches="tight")


def get_more_sparse_di_curve(positive: pd.DataFrame, negative: pd.DataFrame):
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))
    nsamples = [50, 500, 5000]
    axs = axs.flatten()
    for ax, n in zip(axs, nsamples):
        tprs_at_fpr_1 = []
        for model in positive.Model.unique():
            tmp_positive = positive.loc[(positive.Model == model) & (positive.n == n)]
            tmp_negative = negative.loc[(negative.Model == model) & (negative.n == n)]
            fpr, tpr, _ = roc_curve(
                [1] * tmp_positive.shape[0] + [0] * tmp_negative.shape[0],
                list(1 - tmp_positive.pvalue) + list(1 - tmp_negative.pvalue),
            )
            ax.plot(fpr, tpr, label=model, color=MODELS_COLORS[model])
            tprs_at_fpr_1.append(f"{model}: {tpr[np.sum(fpr < 0.01)]:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random", color="black")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim([1e-3, 1.1])
        ax.set_xlim([1e-3, 1.1])
        ax.set_title("|$\mathbf{Q_{sus}}|=$" + str(n))
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            tprs_at_fpr_1 + ["Random: 0.01"],
            loc="lower right",
            title="Model: TPR@FPR=1%",
        )
    plt.savefig(f"{PATH_TO_PLOTS}more_sparse_di_roc_curve.pdf", bbox_inches="tight")


def get_mia_only_curve(data: pd.DataFrame):
    x = len(data.Attack.unique())
    y = len(data.Model.unique())

    fig, axs = plt.subplots(y, x, figsize=(5 * x, 5 * y))
    axs = axs.flatten()
    for ax, (model, attack) in tqdm(
        zip(axs, product(data.Model.unique(), data.Attack.unique())), total=len(axs)
    ):
        for n in data.Size.unique():
            tmp_data = data.loc[
                (data.Attack == attack) & (data.Model == model) & (data.Size == n)
            ]
            fpr, tpr, _ = roc_curve(
                [1] * tmp_data.shape[0] + [0] * tmp_data.shape[0],
                list(1 - tmp_data.Members) + list(1 - tmp_data.Nonmembers),
            )
            ax.plot(fpr, tpr, label=n)
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim([1e-3, 1.1])
        ax.set_xlim([1e-3, 1.1])
        ax.set_title(model + ", " + attack.split("~")[0])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[-1]], [labels[-1]], loc="lower right")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles[:-1],
        labels[:-1],
        loc="lower center",
        ncol=len(data.Size.unique()),
        title="$\mathbf{Q_{sus}}$ size",
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.tight_layout()
    plt.savefig(f"{PATH_TO_PLOTS}mia_roc_curve.pdf", bbox_inches="tight")


def get_final_cmp(data: pd.DataFrame, positive: pd.DataFrame, negative: pd.DataFrame):
    nsamples = [50, 100, 500, 1000, 5000, 10000]
    x = len(nsamples)
    y = len(positive.Model.unique())

    fig, axs = plt.subplots(y, x, figsize=(5 * x, 5 * y))
    axs = axs.flatten()
    for ax, (model, n) in tqdm(
        zip(axs, product(positive.Model.unique(), nsamples)), total=len(axs)
    ):
        tmp_positive = positive.loc[(positive.Model == model) & (positive.n == n)]
        tmp_negative = negative.loc[(negative.Model == model) & (negative.n == n)]
        fpr, tpr, _ = roc_curve(
            [1] * tmp_positive.shape[0] + [0] * tmp_negative.shape[0],
            list(1 - tmp_positive.pvalue) + list(1 - tmp_negative.pvalue),
        )
        ax.plot(fpr, tpr, label=OURS)

        for attack in data.Attack.unique():
            tmp_data = data.loc[
                (data.Attack == attack) & (data.Model == model) & (data.Size == n)
            ]
            fpr, tpr, _ = roc_curve(
                [1] * tmp_data.shape[0] + [0] * tmp_data.shape[0],
                list(1 - tmp_data.Members) + list(1 - tmp_data.Nonmembers),
            )
            ax.plot(fpr, tpr, label=attack)

        ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim([1e-3, 1.1])
        ax.set_xlim([1e-3, 1.1])
        ax.set_title(model + ", |$\mathbf{Q_{sus}}|=$" + str(n))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[-1]], [labels[-1]], loc="lower right")

    handles, labels = axs[0].get_legend_handles_labels()

    labels = [
        label if idx == 0 else label.split("~")[0] for idx, label in enumerate(labels)
    ][:-1]

    fig.legend(
        handles[:-1],
        labels,
        loc="lower center",
        ncol=len(attacks),
        title="Attack",
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.tight_layout()
    plt.savefig(f"{PATH_TO_PLOTS}final_roc_curve.pdf", bbox_inches="tight")


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    positive = pd.read_csv("experiments/plots/pvalue_per_sample/pvalue_per_sample.csv")
    negative = pd.read_csv(
        "experiments/plots/noise_influence/pvalue_per_sample_only_noise.csv"
    )
    data = pd.read_csv("experiments/plots/mia_di/max_scores.csv")

    data = data.loc[(data.Attack != OURS) & (data.Model.isin(MODELS))]
    positive = positive.loc[positive.Attack == OURS]
    negative = negative.loc[negative.Attack == OURS]

    get_di_only_curve(positive, negative)
    get_more_sparse_di_curve(positive, negative)
    get_mia_only_curve(data)

    positive = positive.loc[positive.Model.isin(MODELS)]
    negative = negative.loc[negative.Model.isin(MODELS)]

    get_final_cmp(data, positive, negative)


if __name__ == "__main__":
    main()