from torch import Tensor as T
from src.attacks import DataSource


class ScoreComputer(DataSource):
    def check_data(self, members: T, nonmembers: T) -> None:
        """
        Check that the data is in the correct format
        """
        assert len(members.shape) == len(nonmembers.shape) == 1
        assert members.shape[0] == nonmembers.shape[0] == self.total_samples

    def run(self, members: T, nonmembers: T, *args, **kwargs) -> None:
        """
        Compute scores
        """
        # 1. Collect scores for members and nonmembers
        members_scores, nonmembers_scores = self.process_data(
            members, nonmembers, *args, **kwargs
        )
        # 2. Run assertions
        self.check_data(members_scores, nonmembers_scores)
        # 3. Save scores
        self.save(members_scores, nonmembers_scores)
