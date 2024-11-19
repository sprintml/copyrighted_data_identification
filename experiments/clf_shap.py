import sys
import os
sys.path.append(".")
sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import torch
from torch import Tensor as T
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.attacks import get_datasets_clf
from experiments.utils import (
    set_plt,
    RUN_ID,
    MODELS,
    ATTACKS,
    ATTACKS_NAME_MAPPING,
)

import matplotlib.pyplot as plt
import shap
from PIL import Image
from tqdm import tqdm

from utils import MODELS_NAME_MAPPING


set_plt()

PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "experiments/out/plots/features_ablation"
n_eval_samples = 20_000
members_lower = False

feature_names = (
    [ATTACKS_NAME_MAPPING[attack] for attack in ATTACKS[:4]]
    + [
        ATTACKS_NAME_MAPPING[attack] + "_" + str(idx)
        for attack in ATTACKS[4:6]
        for idx in range(10)
    ]
    + [ATTACKS_NAME_MAPPING[ATTACKS[6]] + "_" + str(idx) for idx in range(2)]
)


def get_datasets(
    members: T,
    nonmembers: T,
):

    members_train = members[:5000]
    members_test = members[5000:]
    nonmembers_train = nonmembers[:5000]
    nonmembers_test = nonmembers[5000:]
    
    # Create training and testing datasets using get_datasets_clf
    train_dataset = get_datasets_clf(members_train, nonmembers_train)
    test_dataset = get_datasets_clf(members_test, nonmembers_test)

    ss = StandardScaler()

    train_dataset.data = ss.fit_transform(train_dataset.data)
    test_dataset.data = ss.transform(test_dataset.data)

    return train_dataset, test_dataset


def get_clf(model: str):
    data = np.load(
        f"{PATH_TO_FEATURES}/{model}_combination_attack_{RUN_ID}.npz", allow_pickle=True
    )
    members = torch.from_numpy(data["members"][:, 0, :])
    nonmembers = torch.from_numpy(data["nonmembers"][:, 0, :])

    train_dataset, test_dataset = get_datasets(
        members,
        nonmembers,
    )

    clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=None, solver="liblinear")
    clf.fit(train_dataset.data, train_dataset.label)
    return clf, train_dataset, test_dataset


def plot_multiple_shap_summaries(models_dict):
    for i, (model_name, (model, _, data)) in tqdm(enumerate(models_dict.items())):
        try:
            Image.open(f"{PATH_TO_PLOTS}/shap_{model_name}.png")
            continue
        except:
            pass
        explainer = shap.LinearExplainer(model, data.data)
        shap_values = explainer(data.data)
        shap.summary_plot(
            shap_values,
            data.data,
            feature_names=feature_names,
            max_display=len(feature_names),
            show=False,
            sort=False,
        )
        fig = plt.gcf()
        fig.savefig(f"{PATH_TO_PLOTS}/shap_{model_name}.png", dpi=300)
        plt.close(fig)

    fig, axs = plt.subplots(
        2, len(models_dict) // 2, figsize=(5 * len(models_dict), 30)
    )
    axs = axs.flatten()

    for ax, (model_name, _) in zip(axs, models_dict.items()):
        ax.imshow(Image.open(f"{PATH_TO_PLOTS}/shap_{model_name}.png"))
        ax.axis("off")
        ax.set_title(MODELS_NAME_MAPPING[model_name])

    plt.tight_layout()
    fig.savefig(f"{PATH_TO_PLOTS}/shap_summary.pdf", dpi=300)


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    clfs = {model: get_clf(model) for model in MODELS}
    plot_multiple_shap_summaries(clfs)


if __name__ == "__main__":
    main()