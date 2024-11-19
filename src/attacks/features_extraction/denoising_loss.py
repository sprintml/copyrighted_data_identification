# https://arxiv.org/abs/2301.13188

from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple


class DenoisingLossExtractor(FeatureExtractor):
    def process_batch(self, batch: Tuple[T, T]) -> T:
        """
        TODO: create docstring
        """
        images, classes = batch
        bs = images.shape[0]
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device)
        latents = self.model.encode(images)

        losses = []
        for _ in range(self.attack_cfg.n_repetitions):
            noise = torch.randn_like(latents).to(self.device)
            noise_pred = self.model.predict_noise_from_latent(
                latents, classes, self.attack_cfg.timestep, noise
            )
            loss = torch.norm(
                noise_pred.reshape(bs, -1) - noise.reshape(bs, -1), dim=-1, p=2
            ).cpu()
            losses.append(loss)

        losses = torch.stack(losses, dim=1).unsqueeze(2)  # B, N_measurements, 1 feature
        return losses
