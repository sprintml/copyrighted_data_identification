from typing import Dict

from src.models.GeneralLatentDiffusionWrapper import (
    GeneralLatentDiffusionWrapper as DiffusionModel,
)
from src.models.CompVisLatentDiffusionWrapper import (
    CompVisLatentDiffusionWrapper as LatentDiffusionModel,
)
from src.models.DiTWrapper import (
    DiTWrapper as DiTModel,
)
from src.models.UViTWrapper import (
    UViTWrapper as UViTModel,
)
from src.models.CIFAR10DDPMWrapper import (
    CIFAR10DDPMWrapper as CIFAR10Model,
)

from src.models.SecmiCIFAR10ModelWrapper import (
    SecmiCIFAR10ModelWrapper as SecmiModel,
)
from src.models.UViT_t2i_Wrapper import (
    UViT_t2i_Wrapper as UViT_t2i_Model,
)
from src.models.UViT_uncond_Wrapper import (
    UViT_uncond_Wrapper as UViT_uncond_Model,
)


diffusion_models: Dict[str, DiffusionModel] = {
    "dummy": DiffusionModel,
    "ldm": LatentDiffusionModel,
    "dit": DiTModel,
    "dit_512": DiTModel,
    "uvit": UViTModel,
    "uvit_512": UViTModel,
    "uvit_t2i": UViT_t2i_Model,
    "uvit_t2i_deep": UViT_t2i_Model,
    "uvit_uncond": UViT_uncond_Model,
    "cifar10": CIFAR10Model,
    "cifar10_secmi": SecmiModel,
}
