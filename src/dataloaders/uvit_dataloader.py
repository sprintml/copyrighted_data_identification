import os
import torch
from typing import Tuple
from torchvision import datasets
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import datasets


def get_uvit_dataloaders(
    config: DictConfig, model_cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Retrun dataloaders matching DiT model
    TODO: enable 512x512
    """
    dataset = datasets.get_dataset("imagenet")
    train_dataset = dataset.get_split(split="train", labeled=True)
    test_dataset = dataset.get_split(split="test", labeled=True)

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    members_loader = DataLoader(
        train_dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    nonmembers_loader = DataLoader(
        test_dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )

    return members_loader, nonmembers_loader
