from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
import os
import numpy as np
from typing import Tuple


class CDIExtractor(FeatureExtractor):
    def load_data(self, attack: str, folder: str) -> Tuple[T, T]:
        """
        Load the data from the file
        """
        path = os.path.join(
            self._strip_hydra_path(folder),
            f"{self.model_cfg.name}_{attack}_{self.config.run_id}.npz",
        )
        data = np.load(path, allow_pickle=True)
        return torch.from_numpy(data["members"]), torch.from_numpy(data["nonmembers"])

    def process_data(self) -> Tuple[T, T]:
        members_features, nonmembers_features = [], []
        if self.attack_cfg.attacks_scores_to_use is not None:
            for attack in self.attack_cfg.attacks_scores_to_use.split(","):
                members, nonmembers = self.load_data(attack, self.config.path_to_scores)
                members_features.append(members.view(-1, 1, 1))
                nonmembers_features.append(nonmembers.view(-1, 1, 1))

        if self.attack_cfg.attacks_features_to_use is not None:
            for attack in self.attack_cfg.attacks_features_to_use.split(","):
                members, nonmembers = self.load_data(
                    attack, self.config.path_to_features
                )
                B = members.shape[0]
                members_features.append(members.view(B, 1, -1))
                nonmembers_features.append(nonmembers.view(B, 1, -1))

        return torch.cat(members_features, dim=2), torch.cat(
            nonmembers_features, dim=2
        )  # B, 1, N_features
