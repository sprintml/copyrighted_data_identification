from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple


class MultipleLossExtractor(FeatureExtractor):
    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch
        bs = images.shape[0]
        images = images.to(self.device)
        classes = classes.to(self.device)
        latents = self.model.encode(images)
        noise = torch.randn_like(latents).to(self.device)

        timesteps = [int(t) for t in self.attack_cfg.timesteps.split(",")]

        losses = []
        for timestep in timesteps:
            noise_pred = self.model.predict_noise_from_latent(
                latents, classes, timestep, noise
            )
            loss = torch.norm(
                noise_pred.reshape(bs, -1) - noise.reshape(bs, -1), dim=-1, p=2
            ).cpu()
            losses.append(loss)

        losses = torch.stack(losses, dim=1).unsqueeze(2)  # B, timesteps, 1 feature
        return losses
