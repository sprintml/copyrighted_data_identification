import torch
from torch import Tensor as T
from src.models import DiffusionModel

from DiT.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from DiT.download import find_model
from DiT.models import DiT_XL_2


class DiTWrapper(DiffusionModel):
    """
    TODO: check vae training details - ema or mse
    """

    def __init__(self, model_cfg):
        super(DiTWrapper, self).__init__(model_cfg)
        self.image_size = self.model_cfg.image_size
        self.latent_size = int(self.image_size) // 8

        self.vae = AutoencoderKL.from_pretrained(
            self.model_cfg.vae_model, cache_dir="./model_checkpoints/dit"
        ).to(self.device)

        self.model = DiT_XL_2(input_size=self.latent_size).to(self.device)
        state_dict = torch.load(self.model_cfg.diffuser_model)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.diffusion = create_diffusion(timestep_respacing="")

    def _encode(self, images: T) -> T:
        """
        Encodes image batch into latents using first stage VAE model
        """
        return self.vae.encode(images).latent_dist.sample().mul_(0.18215)

    def _decode(self, latents: T) -> T:
        """
        Decodes latents batch into images using first stage VAE model
        """
        return self.vae.decode(latents / 0.18215).sample

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        return self.diffusion.q_sample(
            latents,
            torch.ones(latents.shape[0], device=self.device).long() * timestep,
            noise=noise,
        )

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        """
        Predics noise for a noised latent at timestep t
        """
        t = torch.ones(latents_noisy.shape[0], device=self.device).long() * timestep
        model_output = self.model(latents_noisy, t, y=classes)
        # pred noise, pred_variance
        model_output, model_var_values = torch.split(
            model_output, latents_noisy.shape[1], dim=1
        )
        return model_output

    def get_alpha_cumprod(self, t: int) -> float:
        """
        Return cumulative product of alphas for timestep t
        """
        return self.diffusion.alphas_cumprod[t]
