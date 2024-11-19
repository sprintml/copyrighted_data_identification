import os
import torch
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from PIL import Image
import numpy as np


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def get_dit_dataloaders(
    config: DictConfig, model_cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Retrun dataloaders matching DiT model
    TODO: enable 512x512
    """

    # Setup data
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, model_cfg.image_size)
            ),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )

    # Define paths for train and validation data
    train_data_path = os.path.join(model_cfg.dataset_path, "train")
    val_data_path = os.path.join(model_cfg.dataset_path, "val")

    # Create datasets
    members_dataset = datasets.ImageFolder(train_data_path, transform=transform)
    nonmembers_dataset = datasets.ImageFolder(val_data_path, transform=transform)

    # Create data loaders for members and nonmembers
    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    members_loader = DataLoader(
        members_dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    nonmembers_loader = DataLoader(
        nonmembers_dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )

    return members_loader, nonmembers_loader
