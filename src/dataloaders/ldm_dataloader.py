import os
import torch
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig


def get_ldm_dataloaders(
    config: DictConfig, model_cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Retrun dataloaders matching ldm model
    """

    # Define transformations
    convert_to_input = transforms.Compose(
        [
            transforms.Resize(model_cfg.image_size),
            transforms.CenterCrop(model_cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Define paths for train and validation data
    train_data_path = os.path.join(model_cfg.dataset_path, "train")
    val_data_path = os.path.join(model_cfg.dataset_path, "val")

    # Create datasets
    members_dataset = datasets.ImageFolder(train_data_path, transform=convert_to_input)
    nonmembers_dataset = datasets.ImageFolder(val_data_path, transform=convert_to_input)

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
