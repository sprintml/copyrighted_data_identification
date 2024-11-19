import torch
from torch import Tensor as T
from src.models import DiffusionModel

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel


class CIFAR10DDPMWrapper(DiffusionModel):
    def __init__(self, model_cfg):
        super(CIFAR10DDPMWrapper, self).__init__(model_cfg)
        self.image_size = 32

        self.model_name = "google/ddpm-cifar10-32"

        pipeline = DDPMPipeline.from_pretrained(self.model_name)  # TODO add cache dir
        self.model: UNet2DModel = pipeline.unet.to(self.device)
        self.model.eval()
        self.scheduler: DDPMScheduler = pipeline.scheduler

    def _encode(self, images: T) -> T:
        """
        We don't have a VAE model here, so we just return the images
        """
        return images

    def _decode(self, latents: T) -> T:
        """
        We don't have a VAE model here, so we just return the "latents"
        """
        return latents

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        return self.scheduler.add_noise(latents, noise, torch.tensor(timestep))

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        """
        Predics noise for a noised "latent" at timestep t
        """
        model_output = self.model(latents_noisy, timestep).sample

        return model_output

    def get_alpha_cumprod(self, t: int) -> float:
        """
        Return cumulative product of alphas for timestep t
        """
        return self.scheduler.alphas_cumprod[t]
