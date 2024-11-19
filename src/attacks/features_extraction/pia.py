# https://openreview.net/pdf?id=rpH9FcCEV6

from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple, List
from math import sqrt
import os
import numpy as np
from math import pi


class PIAExtractor(FeatureExtractor):
    def normalize(self, eps: T) -> T:
        # Eq. 10
        _, C, H, W = eps.shape
        N = C * H * W

        return N * sqrt(pi / 2) * eps / eps.norm(p=1, dim=(1, 2, 3), keepdim=True)

    def get_eps_zero(self, latents: T, classes: T) -> T:
        return self.model.predict_noise_from_latent(latents, classes, 0)  # B, C, H, W

    def get_eps_t(self, latents: T, eps_zero: T, classes: T, timestep: int) -> T:
        alpha_t = self.model.get_alpha_cumprod(timestep)
        input_latents = sqrt(alpha_t) * latents + sqrt(1 - alpha_t) * eps_zero
        return self.model.predict_noise_from_latent(input_latents, classes, timestep)

    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch
        bs = images.shape[0]
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device)
        latents = self.model.encode(images)

        eps_zero = self.get_eps_zero(latents, classes)

        # Eq. 9
        pia_score = torch.norm(
            eps_zero - self.get_eps_t(latents, eps_zero, classes, self.attack_cfg.t),
            p=self.attack_cfg.p,
            dim=(1, 2, 3),
        ).view(bs, 1, 1)
        eps_zero = self.normalize(eps_zero)
        pian_score = torch.norm(
            eps_zero - self.get_eps_t(latents, eps_zero, classes, self.attack_cfg.t),
            p=self.attack_cfg.p,
            dim=(1, 2, 3),
        ).view(bs, 1, 1)

        return torch.concatenate([pia_score, pian_score], dim=2)  # B, 1, 2

    def process_data(self) -> Tuple[T, T]:
        second_attack = "pian" if self.attack_cfg.name == "pia" else "pia"
        path = os.path.join(
            *self.path_out.split("/")[:-1],
            f"{self.model_cfg.name}_{second_attack}_{self.config.run_id}.npz",
        )
        try:
            data = np.load(path, allow_pickle=True)
            print(f"Reading pre-computed features from {second_attack} attack, {path=}")
            return torch.from_numpy(data["members"]), torch.from_numpy(
                data["nonmembers"]
            )
        except FileNotFoundError:
            pass
        print(f"Computing features for {self.attack_cfg.name} attack")
        return super().process_data()
