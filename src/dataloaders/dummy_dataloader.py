import torch
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig

# Define transformations


def get_dummy_dataloaders(
    config: DictConfig, model_cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Retrun dataloaders matching dummy model
    """
    t = transforms.Compose(
        [
            transforms.Resize(
                (model_cfg.image_size, model_cfg.image_size), antialias=True
            ),
            transforms.ToTensor(),
        ]
    )

    dummy_test_dataset = datasets.CIFAR10(
        root=model_cfg.dataset_path, train=False, download=True, transform=t
    )
    dummy_train_dataset = datasets.CIFAR10(
        root=model_cfg.dataset_path, train=True, download=True, transform=t
    )

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    members_loader = DataLoader(
        dummy_train_dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    nonmembers_loader = DataLoader(
        dummy_test_dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )

    return members_loader, nonmembers_loader
