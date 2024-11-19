from torch import Tensor as T
import torch
import random

from typing import Optional


class GeneralLatentDiffusionWrapper(torch.nn.Module):
    def __init__(self, model_cfg):
        super(GeneralLatentDiffusionWrapper, self).__init__()
        self.model_cfg = model_cfg
        self.device = model_cfg.device

    def _encode(self, images: T) -> T:
        return torch.randn(images.shape[0], 3, 16, 16).to(self.device)

    def encode(self, images: T, use_grad: bool = False) -> T:
        """
        Encodes image batch into latents using first stage VAE model
        If use_grad is False, the function is called in no_grad mode
        """
        if use_grad:
            latents = self._encode(images)
        else:
            with torch.no_grad():
                latents = self._encode(images)
        return latents

    def _decode(self, latents: T) -> T:
        return torch.randn(latents.shape[0], 3, 32, 32).to(self.device)

    def decode(self, latents: T, use_grad: bool = False) -> T:
        """
        Decodes latents batch into images using first stage VAE model
        If use_grad is False, the function is called in no_grad mode
        """
        if use_grad:
            images = self._decode(latents)
        else:
            with torch.no_grad():
                images = self._decode(latents)
        return images

    def get_alpha_cumprod(self, timestep: int) -> float:
        return random.uniform(0, 1)

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        return torch.randn_like(latents_noisy)

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        """
        Applies model-specific noise schedule to latents
        """
        return noise

    def predict_noise_from_latent(
        self,
        latents: T,
        classes: T,
        timestep: int,
        noise: Optional[T] = None,
        use_grad: bool = False,
    ) -> T:
        """
        Predicts noise for a noised latent at timestep t
        If use_grad is False, the function is called in no_grad mode
        If noise is None, the function assumes the input latent is already noised
        """
        if noise is not None:
            latents_noisy = self.noise_latents(latents, timestep, noise)
        else:
            latents_noisy = latents

        if use_grad:
            noise_pred = self._predict_noise_from_latent(
                latents_noisy, classes, timestep
            )
        else:
            with torch.no_grad():
                noise_pred = self._predict_noise_from_latent(
                    latents_noisy, classes, timestep
                )
        return noise_pred

    def generate(self, n: int) -> T:
        return torch.randn(n, 3, 32, 32)
