import os.path
import sys


import torch
import torchvision.datasets
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from omegaconf import DictConfig


class MIACIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, idxs, **kwargs):
        super(MIACIFAR10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR10, self).__getitem__(item)


def load_member_data(
    dataset_root,
    dataset_name,
    batch_size=128,
    member_split_root="./member_splits",
    shuffle=False,
    randaugment=False,
):
    if dataset_name.upper() == "CIFAR10":
        splits = np.load(os.path.join(member_split_root, "CIFAR10_train_ratio0.5.npz"))
        member_idxs = splits["mia_train_idxs"]
        nonmember_idxs = splits["mia_eval_idxs"]
        # load MIA Datasets
        if randaugment:
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandAugment(num_ops=5),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )
        else:
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),
                ]
            )
        member_set = MIACIFAR10(
            member_idxs,
            root=os.path.join(dataset_root, "cifar10"),
            train=True,
            transform=transforms,
        )
        nonmember_set = MIACIFAR10(
            nonmember_idxs,
            root=os.path.join(dataset_root, "cifar10"),
            train=True,
            transform=transforms,
        )
    else:
        raise NotImplemented

    member_loader = torch.utils.data.DataLoader(
        member_set, batch_size=batch_size, shuffle=shuffle
    )
    nonmember_loader = torch.utils.data.DataLoader(
        nonmember_set, batch_size=batch_size, shuffle=shuffle
    )
    return member_loader, nonmember_loader


def get_cifar10_secmi_dataloaders(
    config: DictConfig, model_cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    return load_member_data(
        dataset_root="dataset_inference_dm/SecMI/datasets",
        dataset_name="CIFAR10",
        member_split_root="dataset_inference_dm/SecMI/mia_evals/member_splits",
    )


def get_cifar10_secmi_dataloaders(
    config: DictConfig, model_cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    splits = np.load(model_cfg.split_path)
    member_idxs = splits["mia_train_idxs"]
    nonmember_idxs = splits["mia_eval_idxs"]

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    member_set = MIACIFAR10(
        member_idxs,
        root=model_cfg.dataset_path,
        train=True,
        download=True,
        transform=transforms,
    )
    nonmember_set = MIACIFAR10(
        nonmember_idxs,
        root=model_cfg.dataset_path,
        train=True,
        download=True,
        transform=transforms,
    )

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)

    members_loader = DataLoader(
        member_set,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    nonmembers_loader = DataLoader(
        nonmember_set,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    return members_loader, nonmembers_loader
