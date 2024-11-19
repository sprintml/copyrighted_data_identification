from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple
from scipy.optimize import minimize
from src.models import DiffusionModel


class NoiseOptimExtractor(FeatureExtractor):
    @staticmethod
    def obj(
        perturbations: T,
        B: int,
        latents_noised: T,
        classes: T,
        timestep: int,
        noise: T,
        model: DiffusionModel,
        device: str,
    ) -> Tuple[float, T]:
        perturbations = torch.from_numpy(perturbations).requires_grad_(True)
        input_latents = latents_noised + perturbations.reshape(*latents_noised.shape)
        input_latents = input_latents.to(device).float()
        noise = noise.to(device).float()
        if type(classes) == T:
            classes = classes.to(device)

        noise_pred = model.predict_noise_from_latent(
            input_latents, classes, timestep, use_grad=True
        )
        loss = torch.norm(
            noise_pred.reshape(B, -1) - noise.reshape(B, -1), dim=-1, p=2
        ).mean()
        loss.backward()
        gradient = perturbations.grad.view(-1)

        return loss.item(), gradient.cpu()

    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch
        bs = images.shape[0]
        images = images.to(self.device)
        latents = self.model.encode(images)

        noise = torch.randn_like(latents).to(self.device)

        latents_noised = self.model.noise_latents(
            latents,
            self.attack_cfg.timestep,
            noise,
        )

        out = minimize(
            self.obj,
            torch.zeros_like(latents_noised).cpu().reshape(-1),
            args=(
                bs,
                latents_noised.cpu(),
                classes.cpu(),
                self.attack_cfg.timestep,
                noise.cpu(),
                self.model,
                self.device,
            ),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.attack_cfg.max_iter},
        )

        perturbations = torch.from_numpy(out.x.reshape(*latents_noised.shape)).to(
            self.device
        )

        noise_pred = self.model.predict_noise_from_latent(
            (latents_noised + perturbations).float(),
            classes.to(self.device),
            self.attack_cfg.timestep,
            noise,
        )
        loss = torch.norm(
            noise_pred.reshape(bs, -1) - noise.reshape(bs, -1), dim=-1, p=2
        ).cpu()

        l2_perturbations = torch.norm(perturbations.view(bs, -1), dim=-1, p=2).cpu()

        return torch.stack([loss, l2_perturbations], dim=1).unsqueeze(
            1
        )  # B, 1, 2 features
