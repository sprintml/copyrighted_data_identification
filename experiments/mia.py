import os
import sys

sys.path.append(".")
sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from itertools import product

from src import get_tpr_fpr, get_accuracy, get_auc
from src.evaluation.evaluate import scale_and_order, get_y_true_y_score
from sklearn.metrics import roc_curve
from experiments.utils import (
    set_plt,
    ATTACKS_NAME_MAPPING,
    ATTACKS_COLORS,
    MODELS_NAME_MAPPING,
    MODELS_ORDER,
    MODELS,
    RUN_ID,
    MIAS_CITATIONS,
    ATTACKS,
    OURS,
    RESAMPLING_CNT,
)

from typing import Tuple, List

set_plt()

SIZE = 10_000
PATH_TO_SCORES = "out/scores"
PATH_TO_PLOTS = "experiments/out/mia_performance"

ATTACKS_ORDER = [ATTACKS_NAME_MAPPING[attack] for attack in ATTACKS]
ATTACKS_ORDER = [attack + MIAS_CITATIONS.get(attack, "") for attack in ATTACKS_ORDER]


def get_data() -> List:
    out = []
    for attack, model in tqdm(product(ATTACKS, MODELS)):

        data = np.load(
            f"{PATH_TO_SCORES}/{model}_{attack}_{RUN_ID}.npz", allow_pickle=True
        )
        members_lower = data["metadata"][()]["members_lower"]
        n_eval_samples = data["metadata"][()]["n_samples_eval"]
        members = data["members"][:n_eval_samples]
        nonmembers = data["nonmembers"][:n_eval_samples]

        attack_mapped = ATTACKS_NAME_MAPPING[attack]
        attack = attack_mapped + MIAS_CITATIONS.get(attack_mapped, "")
        model = MODELS_NAME_MAPPING[model]

        for _ in range(RESAMPLING_CNT):
            indices = np.random.permutation(n_eval_samples)[:SIZE]

            tpr = get_tpr_fpr(
                members[indices],
                nonmembers[indices],
                fpr_threshold=0.01,
                members_lower=members_lower,
            )
            auc = get_auc(
                members[indices], nonmembers[indices], members_lower=members_lower
            )
            accuracy = get_accuracy(
                members[indices], nonmembers[indices], members_lower=members_lower
            )

            out.append(
                [
                    attack,
                    model,
                    f"{tpr*100:.2f}",
                    f"{auc*100:.2f}",
                    f"{accuracy*100:.2f}",
                ]
            )

    return out


def get_roc_data() -> List:
    roc = []
    for attack, model in tqdm(product(ATTACKS, MODELS)):
        data = np.load(
            f"{PATH_TO_SCORES}/{model}_{attack}_{RUN_ID}.npz", allow_pickle=True
        )

        members_lower = data["metadata"][()]["members_lower"]
        n_eval_samples = data["metadata"][()]["n_samples_eval"]
        members = data["members"][:n_eval_samples]
        nonmembers = data["nonmembers"][:n_eval_samples]

        attack_mapped = ATTACKS_NAME_MAPPING[attack]
        attack = attack_mapped + MIAS_CITATIONS.get(attack_mapped, "")
        model = MODELS_NAME_MAPPING[model]
        members_scores, nonmembers_scores = scale_and_order(
            members, nonmembers, members_lower
        )

        y_true, y_score = get_y_true_y_score(members_scores, nonmembers_scores)

        fprs, tprs, _ = roc_curve(y_true, y_score)

        roc.append([attack_mapped, model, fprs, tprs])

    return roc


def plot_rocs(roc_data: List):
    x = 4
    y = len(MODELS) // x + (1 if len(MODELS) % x != 0 else 0)

    fig, axs = plt.subplots(y, x, figsize=(x * 7, y * 7))
    axs = axs.flatten()
    for attack, model, fprs, tprs in tqdm(roc_data):
        idx = MODELS_ORDER.index(model)
        ax: plt.Axes = axs[idx]
        ax.plot(fprs, tprs, label=attack, color=ATTACKS_COLORS[attack])
        ax.set(
            title=f"{model}",
            xlabel="FPR" if not idx else "",
            ylabel="TPR" if not idx else "",
            xlim=[1e-3, 1],
            ylim=[1e-3, 1],
            yscale="log",
            xscale="log",
        )

    for i in range(len(axs)):
        axs[i].plot([1e-3, 1], [1e-3, 1], "--", color="black", label="Random")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(ATTACKS) // 2 + 1,
        title="Attack",
    )
    plt.savefig(f"{PATH_TO_PLOTS}/mia_rocs.pdf", bbox_inches="tight")


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    np.random.seed(42)

    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/mia_performance.csv")
    except FileNotFoundError:
        data = get_data()
        df = pd.DataFrame(
            data, columns=["Attack", "Model", "TPR@FPR=1\%", "AUC", "Accuracy"]
        )
        df.to_csv(f"{PATH_TO_PLOTS}/mia_performance.csv")

    roc_data = get_roc_data()

    attacks_main_paper = [
        attack + citation for attack, citation in MIAS_CITATIONS.items()
    ] + [OURS]

    for name, col_name in zip(
        ["tpr", "auc", "acc"], ["TPR@FPR=1\%", "AUC", "Accuracy"]
    ):
        for post, attacks in zip(
            ["main", "appendix"], [attacks_main_paper, ATTACKS_ORDER]
        ):
            if post == "main" and name != "tpr":
                continue
            tmp_df = (
                df.groupby(["Model", "Attack"])[col_name]
                .apply(
                    lambda x: "{mean:.2f}{{\\tiny $\pm${std:.2f}}}".format(
                        mean=x.mean(),
                        std=x.std(),
                    )
                )
                .reset_index()
                .pivot(index="Model", columns="Attack", values=col_name)
                .loc[MODELS_ORDER][attacks]
                .T
            )

            tmp_df.to_latex(
                f"{PATH_TO_PLOTS}/mia_performance_{name}_{post}.tex", escape=False
            )

    plot_rocs(roc_data)


if __name__ == "__main__":
    main()
