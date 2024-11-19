import torch
from torch.utils.data import Dataset
from torch import Tensor as T

from typing import Tuple

from src.attacks import DataSource
from typing import Optional


class MIDataset(Dataset):
    def __init__(
        self,
        member_data: T,
        nonmember_data: T,
    ):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat(
            [torch.ones(member_data.size(0)), torch.zeros(nonmember_data.size(0))]
        )

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def load_data(
    source: DataSource,
    directory: str,
    n_samples: Optional[int] = None,
    override_filename: Optional[str] = None,
) -> Tuple[T, T, dict]:
    """
    Load data, features or scores
    """
    members_scores, nonmembers_scores, metadata = source.load(
        directory=directory, override_filename=override_filename
    )
    return members_scores[:n_samples], nonmembers_scores[:n_samples], metadata


def get_datasets_clf(
    members: T, nonmembers: T
) -> Tuple[MIDataset]:
    dataset = MIDataset(members, nonmembers)
    return dataset