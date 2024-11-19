import torch
from torch import Tensor as T
import sys
from absl import flags

from src.models import DiffusionModel
from SecMI.model import UNet
from SecMI.diffusion import GaussianDiffusionTrainer


def get_FLAGS(flag_path):
    FLAGS = flags.FLAGS
    flags.DEFINE_bool("train", False, help="train from scratch")
    flags.DEFINE_bool("eval", False, help="load ckpt.pt and evaluate FID and IS")
    # UNet
    flags.DEFINE_integer("ch", 128, help="base channel of UNet")
    flags.DEFINE_multi_integer("ch_mult", [1, 2, 2, 2], help="channel multiplier")
    flags.DEFINE_multi_integer("attn", [1], help="add attention to these levels")
    flags.DEFINE_integer("num_res_blocks", 2, help="# resblock in each level")
    flags.DEFINE_float("dropout", 0.1, help="dropout rate of resblock")
    # Gaussian Diffusion
    flags.DEFINE_float("beta_1", 1e-4, help="start beta value")
    flags.DEFINE_float("beta_T", 0.02, help="end beta value")
    flags.DEFINE_integer("T", 1000, help="total diffusion steps")
    flags.DEFINE_enum(
        "mean_type", "epsilon", ["xprev", "xstart", "epsilon"], help="predict variable"
    )
    flags.DEFINE_enum(
        "var_type", "fixedlarge", ["fixedlarge", "fixedsmall"], help="variance type"
    )
    # Training
    flags.DEFINE_float("lr", 2e-4, help="target learning rate")
    flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
    flags.DEFINE_integer("total_steps", 800000, help="total training steps")
    flags.DEFINE_integer("img_size", 32, help="image size")
    flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
    flags.DEFINE_integer("batch_size", 128, help="batch size")
    flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
    flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
    flags.DEFINE_bool("parallel", False, help="multi gpu training")
    # Logging & Sampling
    flags.DEFINE_string("logdir", "./logs/DDPM_CIFAR10_EPS", help="log directory")
    flags.DEFINE_integer("sample_size", 64, "sampling size of images")
    flags.DEFINE_integer("sample_step", 1000, help="frequency of sampling")
    # Evaluation
    flags.DEFINE_integer(
        "save_step",
        80000,
        help="frequency of saving checkpoints, 0 to disable during training",
    )
    flags.DEFINE_integer(
        "eval_step",
        0,
        help="frequency of evaluating model, 0 to disable during training",
    )
    flags.DEFINE_integer(
        "num_images", 50000, help="the number of generated images for evaluation"
    )
    flags.DEFINE_bool("fid_use_torch", False, help="calculate IS and FID on gpu")
    flags.DEFINE_string("fid_cache", "./stats/cifar10.train.npz", help="FID cache")

    FLAGS.read_flags_from_files(flag_path)
    return FLAGS


def get_model(ckpt, FLAGS, WA=True):
    model = UNet(
        T=FLAGS.T,
        ch=FLAGS.ch,
        ch_mult=FLAGS.ch_mult,
        attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
    )
    # load model and evaluate
    ckpt = torch.load(ckpt)

    if WA:
        weights = ckpt["ema_model"]
    else:
        weights = ckpt["net_model"]

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith("module."):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class SecmiCIFAR10ModelWrapper(DiffusionModel):
    def __init__(self, model_cfg):
        super(SecmiCIFAR10ModelWrapper, self).__init__(model_cfg)

        FLAGS = get_FLAGS(self.model_cfg.model_cfg_path)
        FLAGS(sys.argv)
        self.FLAGS = FLAGS
        self.model = get_model(self.model_cfg.diffuser_model, FLAGS, WA=True).to(
            self.device
        )
        self.diffusion = GaussianDiffusionTrainer(
            self.model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T
        ).to(self.device)

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
        """ """

        t = torch.ones(latents.shape[0], device=self.device).long() * timestep
        noise = torch.randn_like(latents)
        noised_latents = (
            extract(self.diffusion.sqrt_alphas_bar, t, latents.shape) * latents
            + extract(self.diffusion.sqrt_one_minus_alphas_bar, t, latents.shape)
            * noise
        )
        return noised_latents

    def _predict_noise_from_latent(
        self, latents_noisy: T, classes: T, timestep: int
    ) -> T:
        """
        Predics noise for a noised "latent" at timestep t
        """
        timestep = (
            torch.ones(latents_noisy.shape[0], device=self.device).long() * timestep
        )
        model_output = self.model(latents_noisy, timestep)
        return model_output

    def get_alpha_cumprod(self, t: int) -> float:
        """
        Return cumulative product of alphas for timestep t
        """
        # return self.diffusion.sqrt_alphas_bar[t] ** 2
        betas = (
            torch.linspace(self.FLAGS.beta_1, self.FLAGS.beta_T, self.FLAGS.T)
            .double()
            .to(self.device)
        )
        alphas = 1.0 - betas
        alphas = torch.cumprod(alphas, dim=0)
        return alphas[t]
