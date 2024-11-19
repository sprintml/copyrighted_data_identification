# https://arxiv.org/pdf/2302.01316.pdf

from src.attacks import ScoreComputer
from src.attacks.utils import get_datasets_clf
from torch import Tensor as T
import torch
from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet

from typing import Tuple
from tqdm import tqdm
import copy


class PIAComputer(ScoreComputer):
    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:
        """
        Inputs of shape (N_samples, 1, 2)
        Outputs of shape (N_samples,)
        """
        return members[:, 0, 0], nonmembers[:, 0, 0]


class PIANComputer(ScoreComputer):
    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:
        """
        Inputs of shape (N_samples, 1, 2)
        Outputs of shape (N_samples,)
        """
        return members[:, 0, 1], nonmembers[:, 0, 1]
