import sys
import os

sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")


import hydra
import torch
import random
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict
from itertools import product

from src import (
    get_tpr_fpr,
    get_p_value,
    get_accuracy,
    get_auc,
    load_data,
    DataSource,
    diffusion_models,
    feature_extractors,
    score_computers,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model_cfg = cfg.model
    action_cfg = cfg.action
    attack_cfg = cfg.attack
    config = cfg.cfg
    with open_dict(model_cfg):
        model_cfg.device = action_cfg.device
        model_cfg.seed = config.seed

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    if action_cfg.name == "features_extraction":
        features_extraction(config, model_cfg, action_cfg, attack_cfg)
    elif action_cfg.name == "scores_computation":
        scores_computation(config, model_cfg, action_cfg, attack_cfg)
    elif action_cfg.name == "evaluation":
        evaluation(config, model_cfg, action_cfg, attack_cfg)
    elif action_cfg.name == "evaluation_bulk":
        evaluation_bulk(config, model_cfg, action_cfg, attack_cfg)
    else:
        raise ValueError("Invalid action name")

    print("fin")


def features_extraction(
    config: DictConfig,
    model_cfg: DictConfig,
    action_cfg: DictConfig,
    attack_cfg: DictConfig,
) -> None:
    model = diffusion_models[model_cfg.name](model_cfg)
    extractor = feature_extractors[attack_cfg.name](
        config, model_cfg, action_cfg, attack_cfg, model
    )
    extractor.run()


def scores_computation(
    config: DictConfig,
    model_cfg: DictConfig,
    action_cfg: DictConfig,
    attack_cfg: DictConfig,
) -> None:
    computer = score_computers[attack_cfg.name](
        config, model_cfg, action_cfg, attack_cfg
    )
    members_features, nonmembers_features, _ = load_data(
        computer, config.path_to_features
    )
    computer.run(members_features, nonmembers_features)


def evaluation(
    config: DictConfig,
    model_cfg: DictConfig,
    action_cfg: DictConfig,
    attack_cfg: DictConfig,
) -> None:
    members_scores, nonmembers_scores, _ = load_data(
        DataSource(config, model_cfg, action_cfg, attack_cfg),
        config.path_to_scores,
        n_samples=config.n_samples_eval,
    )
    if action_cfg.score == "tpr@fpr":
        tpr_01 = get_tpr_fpr(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            fpr_threshold=0.001,
            members_lower=attack_cfg.members_lower,
        )
        print(f"TPR at FPR=0.1%: {tpr_01}")

        tpr_1 = get_tpr_fpr(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            fpr_threshold=0.01,
            members_lower=attack_cfg.members_lower,
        )
        print(f"TPR at FPR=1%: {tpr_1}")

    elif action_cfg.score == "pvalue":
        p_value, is_correct_order = get_p_value(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            members_lower=attack_cfg.members_lower,
        )
        print(f"{p_value=}, {is_correct_order=}")
    elif action_cfg.score == "accuracy":
        accuracy = get_accuracy(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            members_lower=attack_cfg.members_lower,
        )
        print(f"Accuracy: {accuracy}")
    elif action_cfg.score == "auc":
        auc = get_auc(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            members_lower=attack_cfg.members_lower,
        )
        print(f"AUC: {auc}")
    else:
        raise ValueError("Invalid score name")


def evaluation_bulk(
    config: DictConfig,
    model_cfg: DictConfig,
    action_cfg: DictConfig,
    attack_cfg: DictConfig,
) -> None:
    out = {
        "model": [],
        "attack": [],
        "run_id": [],
        "tpr": [],
        "pvalue": [],
        "is_correct_order": [],
        "accuracy": [],
        "auc": [],
    }
    mockup_source = DataSource(config, model_cfg, action_cfg, attack_cfg)
    for idx, (attack, model) in enumerate(
        product(action_cfg.attacks.split(","), action_cfg.models.split(","))
    ):
        filename = f"{model}_{attack}_{config.run_id}"
        members_scores, nonmembers_scores, metadata = load_data(
            mockup_source,
            config.path_to_scores,
            n_samples=config.n_samples_eval,
            override_filename=filename,
        )
        if not idx:
            print(metadata)

        tpr = get_tpr_fpr(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            fpr_threshold=0.01,
            members_lower=metadata["members_lower"],
        )
        p_value, is_correct_order = get_p_value(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            members_lower=metadata["members_lower"],
        )
        accuracy = get_accuracy(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            members_lower=metadata["members_lower"],
        )
        auc = get_auc(
            members_scores=members_scores.numpy(),
            nonmembers_scores=nonmembers_scores.numpy(),
            members_lower=metadata["members_lower"],
        )

        out["model"].append(model)
        out["attack"].append(attack)
        out["run_id"].append(config.run_id)
        out["tpr"].append(tpr)
        out["pvalue"].append(p_value)
        out["is_correct_order"].append(is_correct_order)
        out["accuracy"].append(accuracy)
        out["auc"].append(auc)

    os.makedirs(config.path_to_eval_results, exist_ok=True)
    df = pd.DataFrame(out)
    df.to_csv(
        os.path.join(
            config.path_to_eval_results,
            f"{config.run_id}_{action_cfg.attacks}_{action_cfg.models}.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
