import torch
from torch import Tensor as T
from omegaconf import OmegaConf
from src.models import DiffusionModel
from ldm.util import instantiate_from_config


class CompVisLatentDiffusionWrapper(DiffusionModel):
    def __init__(self, model_cfg):
        super(CompVisLatentDiffusionWrapper, self).__init__(model_cfg)
        self.config = OmegaConf.load(self.model_cfg.model_cfg_path)
        self.model = instantiate_from_config(self.config.model).to(self.device)

    def _encode(self, images: T) -> T:
        """
        Encodes image batch into latents using first stage VAE model
        """
        return self.model.first_stage_model.encode(images)

    def _decode(self, latents: T) -> T:
        """
        Decodes latents batch into images using first stage VAE model
        """
        return self.model.first_stage_model.decode(latents)

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        return self.model.q_sample(
            x_start=latents,
            t=torch.ones(latents.shape[0], device=self.device).long() * timestep,
            noise=noise,
        )

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        """
        Predics noise for a noised latent at timestep t
        """
        c = {"class_label": classes}
        c = self.model.get_learned_conditioning(c)
        t = torch.ones(latents_noisy.shape[0], device=self.device).long() * timestep
        model_output = self.model.apply_model(latents_noisy, t, c)
        return model_output

    def get_alpha_cumprod(self, t: int) -> float:
        """
        Return cumulative product of alphas for timestep t
        """
        return self.model.alphas_cumprod[t]
