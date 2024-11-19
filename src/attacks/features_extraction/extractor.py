import torch
from torch import Tensor as T
from typing import Tuple
from tqdm import tqdm
from src.attacks import DataSource
from src.models import DiffusionModel
from src.dataloaders import loaders


class FeatureExtractor(DataSource):
    model: DiffusionModel

    def process_batch(self, batch: Tuple[T, T], *args, **kwargs) -> T:
        ...

    def process_data(self, *args, **kwargs) -> Tuple[T, T]:
        assert self.model is not None
        members_loader, nonmembers_loader = loaders[self.model_cfg.dataloader](
            self.config, self.model_cfg
        )
        members_features, nonmembers_features = [], []

        samples_processed = 0
        for members_batch, nonmembers_batch in tqdm(
            zip(members_loader, nonmembers_loader)
        ):
            members_features.append(self.process_batch(members_batch, *args, **kwargs))
            nonmembers_features.append(
                self.process_batch(nonmembers_batch, *args, **kwargs)
            )
            samples_processed += members_batch[0].shape[0]
            if samples_processed >= self.total_samples:
                break
        return (
            torch.cat(members_features, dim=0)[: self.total_samples],
            torch.cat(nonmembers_features, dim=0)[: self.total_samples],
        )

    def check_data(self, members: T, nonmembers: T) -> None:
        """
        Check that the data is in the correct format
        """
        assert len(members.shape) == len(nonmembers.shape)
        for i in range(len(members.shape)):
            assert members.shape[i] == nonmembers.shape[i]
        assert len(members.shape) >= 3  # N_samples, N_measurements, *Features
        assert members.shape[0] == self.total_samples

    def run(self, *args, **kwargs) -> None:
        """
        Run the feature extractor
        """
        # 1. Collect features for members and nonmembers
        members_features, nonmembers_features = self.process_data(*args, **kwargs)
        # 2. Run assertions
        self.check_data(members_features, nonmembers_features)
        # 3. Save features
        self.save(members_features, nonmembers_features)
