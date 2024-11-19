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


class SecMIStat(ScoreComputer):
    def compute_score(self, data: T) -> T:
        """
        Compute the score
        Output of shape (N_samples,)
        """
        x_det, x_step = data.permute(1, 0, 2, 3, 4)
        n = x_det.shape[0]
        return torch.norm(x_det.reshape(n, -1) - x_step.reshape(n, -1), dim=-1, p=2)

    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:
        """
        Compute scores
        Inputs of shape (N_samples, 2, C, H, W)
        Outputs of shape (N_samples,)
        """
        return self.compute_score(members), self.compute_score(nonmembers)
