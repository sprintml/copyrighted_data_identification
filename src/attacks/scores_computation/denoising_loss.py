# https://arxiv.org/abs/2301.13188

from src.attacks import ScoreComputer
from torch import Tensor as T

from typing import Tuple


class DenoisingLossComputer(ScoreComputer):
    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:
        """
        Compute scores
        Inputs of shape (N_samples, N_measurements, 1)
        """
        return members.mean(dim=1).squeeze(1), nonmembers.mean(dim=1).squeeze(1)
