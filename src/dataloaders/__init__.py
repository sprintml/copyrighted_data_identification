from src.dataloaders.ldm_dataloader import get_ldm_dataloaders
from src.dataloaders.dummy_dataloader import get_dummy_dataloaders
from src.dataloaders.dit_dataloader import get_dit_dataloaders
from src.dataloaders.uvit_dataloader import get_uvit_dataloaders
from src.dataloaders.uvit_t2i_dataloader import get_uvit_t2i_dataloaders
from src.dataloaders.cifar10_secmi_dataloader import get_cifar10_secmi_dataloaders

from torch.utils.data import DataLoader
from typing import Dict


loaders: Dict[str, DataLoader] = {
    "cifar10": get_dummy_dataloaders,
    "imagenet_compvis": get_ldm_dataloaders,
    "imagenet_dit": get_dit_dataloaders,
    "imagenet_uvit": get_dit_dataloaders,
    "coco_uvit": get_uvit_t2i_dataloaders,
    "cifar10_secmi": get_cifar10_secmi_dataloaders,
}
