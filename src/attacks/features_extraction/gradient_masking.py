from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from math import sqrt
from typing import Tuple, List


class GradientMaskingExtractor(FeatureExtractor):
    def obtain_gradients(
        self, latents: T, classes: T, noise: T, timesteps: List[int]
    ) -> T:
        out = []
        for timestep in timesteps:
            inputs = (
                self.model.noise_latents(latents, timestep, noise)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            noise_pred = self.model.predict_noise_from_latent(
                inputs, classes, timestep, use_grad=True
            )
            loss: T = torch.norm(noise_pred - noise, dim=(1, 2, 3), p=2).mean()
            loss.backward()
            out.append(inputs.grad)
        return torch.stack(out, dim=1)  # B, timesteps, C, H, W

    def get_masks(self, gradients: T, B: int) -> T:
        out = []
        k = int(len(gradients[0][0].flatten()) * self.attack_cfg.masking_ratio)
        for gradient in gradients.permute(1, 0, 2, 3, 4):  # iterate over timesteps
            _, i = torch.topk(gradient.view(B, -1).abs(), k=k, dim=1)
            indices = torch.stack(torch.unravel_index(i, gradient[0].shape)).cpu()
            indices = torch.concatenate(
                [
                    torch.arange(B).repeat_interleave(k).unsqueeze(0),
                    indices.view(len(gradient.shape) - 1, B * k),
                ],
                dim=0,
            )
            mask = torch.zeros_like(gradient)
            mask[list(indices)] = 1
            out.append(mask.bool())

        return torch.stack(out, dim=1)  # B, timesteps, C, H, W

    def noise_top_values(self, latents: T, masks: T, noise: T) -> T:
        out = []
        for mask in masks.permute(1, 0, 2, 3, 4):  # iterate over timesteps:
            inputs = latents.clone().detach()
            inputs[mask] = noise[mask]
            out.append(inputs)

        return torch.stack(out, dim=1)  # B, timesteps, C, H, W

    def masked_noise_pred(
        self, latents_noised: T, classes: T, timesteps: List[int]
    ) -> T:
        out = []
        for idx, timestep in enumerate(timesteps):
            noise_pred = self.model.predict_noise_from_latent(
                latents_noised[:, idx], classes, timestep
            )
            out.append(noise_pred)

        return torch.stack(out, dim=1)  # B, timesteps, C, H, W

    def get_final_loss(
        self, masks: T, latents: T, noise: T, noise_preds: T, B: int
    ) -> T:
        out = []
        for mask, noise_pred in zip(
            masks.permute(1, 0, 2, 3, 4), noise_preds.permute(1, 0, 2, 3, 4)
        ):
            diff = ((noise - latents)[mask] - noise_pred[mask]).reshape(B, -1)
            out.append(torch.norm(diff, dim=1, p=2).unsqueeze(1))

        return torch.stack(out, dim=1)  # B, timesteps, 1

    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch
        bs = images.shape[0]
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device)
        latents = self.model.encode(images)

        noise = torch.randn_like(latents).to(self.device)

        timesteps = [int(t) for t in self.attack_cfg.timesteps.split(",")]

        gradients = self.obtain_gradients(latents, classes, noise, timesteps)
        masks = self.get_masks(gradients, bs)
        latents_noised = self.noise_top_values(latents, masks, noise)
        noise_preds = self.masked_noise_pred(latents_noised, classes, timesteps)

        return self.get_final_loss(
            masks, latents, noise, noise_preds, bs
        )  # B, timesteps, 1
