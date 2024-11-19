import torch
from torch import Tensor as T
from src.models import DiffusionModel

import libs.autoencoder
from libs.uvit_t2i import UViT as UViT_t2i
import numpy as np


def stable_diffusion_beta_schedule(
    linear_start=0.00085, linear_end=0.0120, n_timestep=1000
):
    """
    stable_diffusion_beta_schedule from https://github.com/baofff/U-ViT/blob/main/train_ldm_discrete.py#L23
    """
    _betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    )
    return _betas.numpy()


def stp(s, ts: torch.Tensor):  # scalar tensor product
    """
    Scalar tensor product from https://github.com/baofff/U-ViT/blob/main/train_ldm_discrete.py#L42
    """
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def get_skip(alphas, betas):
    """
    From https://github.com/baofff/U-ViT/blob/main/train_ldm_discrete.py#L30
    """
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1 :] = alphas[s + 1 :].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1 : t + 1] * skip_alphas[1 : t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


class Schedule(object):  # discrete time
    """
    Discrite (normal - timestep is an int) schedule from https://github.com/baofff/U-ViT/blob/main/train_ldm_discrete.py#L53
    """

    def __init__(self, _betas):
        r"""_betas[0...999] = betas[1...1000]
        for n>=1, betas[n] is the variance of q(xn|xn-1)
        for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0.0, _betas)
        self.alphas = 1.0 - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(
        self, latents: T, timestep: np.int8, noise: T
    ) -> T:  # sample from q(xn|x0), where n is uniform
        latents_noisy = stp(self.cum_alphas[timestep] ** 0.5, latents) + stp(
            self.cum_betas[timestep] ** 0.5, noise
        )
        return latents_noisy

    def __repr__(self):
        return f"Schedule({self.betas[:10]}..., {self.N})"


class UViT_t2i_Wrapper(DiffusionModel):
    """
    TODO: check vae training details - ema or mse
    """

    def __init__(self, model_cfg):
        super(UViT_t2i_Wrapper, self).__init__(model_cfg)

        self.image_size = self.model_cfg.image_size
        self.latent_size = self.image_size // 8
        self.patch_size = 2 if self.image_size == 256 else 4

        self.uvit = UViT_t2i(
            img_size=self.latent_size,
            patch_size=self.patch_size,
            in_chans=4,
            embed_dim=512,
            depth=self.model_cfg.depth,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=False,
            mlp_time_embed=False,
            clip_dim=768,
            num_clip_token=77,
        )

        self.uvit.to(self.device)
        self.uvit.load_state_dict(
            torch.load(model_cfg.diffuser_model, map_location="cpu")
        )
        self.uvit.eval()

        self.autoencoder = libs.autoencoder.get_model(model_cfg.vae_model)
        self.autoencoder.scale_factor = 0.23010
        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        betas = stable_diffusion_beta_schedule()
        self.schedule = Schedule(betas)

    def _encode(self, images: T) -> T:
        """
        Encodes image batch into latents using first stage VAE model
        """
        return self.autoencoder.encode(images)

    def _decode(self, latents: T) -> T:
        """
        Decodes latents batch into images using first stage VAE model
        """
        return self.autoencoder.decode(latents)

    def noise_latents(self, latents: T, timestep: int, noise: T) -> T:
        return self.schedule.sample(
            latents, np.ones((latents.shape[0]), dtype=np.int8) * timestep, noise
        )

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        """
        Predics noise for a noised latent at timestep t
        """
        timestep = torch.tensor(
            np.ones((latents_noisy.shape[0]), dtype=np.int8) * timestep,
            device=self.device,
        )
        noise_pred = self.uvit(latents_noisy, timestep, context=classes)
        return noise_pred

    def get_alpha_cumprod(self, t: int) -> float:
        """
        Return cumulative product of alphas for timestep t
        """
        return self.schedule.cum_alphas[t]
