from src.attacks import ScoreComputer
from torch import Tensor as T
import torch


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.attacks.utils import MIDataset, get_datasets_clf
from sklearn.model_selection import KFold

from typing import Tuple

classifiers = {
    "lr": LogisticRegression,
}


class CDIComputer(ScoreComputer):
    def fit_clf(self, train_dataset: MIDataset, seed: int):
        clf = classifiers[self.attack_cfg.clf](
            random_state=seed, **self.attack_cfg.kwargs
        )
        clf.fit(train_dataset["data"], train_dataset["label"])
        return clf

    def compute_score(self, data: T, clf) -> T:
        return torch.from_numpy(clf.predict_proba(data)[:, 1])

    def process_data(self, members: T, nonmembers: T) -> Tuple[T, T]:

        members_indices = np.arange(len(members))
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        members_scores, nonmembers_scores = [], []

        for train_idx, test_idx in kf.split(members_indices):
            # Get train and test splits for members and nonmembers using the same indices
            members_train = members[train_idx]
            members_test = members[test_idx]
            nonmembers_train = nonmembers[train_idx]
            nonmembers_test = nonmembers[test_idx]
            
            # Create training and testing datasets using get_datasets_clf
            train_dataset = get_datasets_clf(members_train, nonmembers_train)
            test_dataset = get_datasets_clf(members_test, nonmembers_test)
            
            # Standardize the data
            ss = StandardScaler()
            train_dataset.data = torch.from_numpy(ss.fit_transform(train_dataset.data))
            test_dataset.data = torch.from_numpy(ss.transform(test_dataset.data))
            
            # Train the classifier
            clf = self.fit_clf(train_dataset, self.config.seed)
            
            # Compute scores only for the test dataset
            scores = self.compute_score(test_dataset.data, clf)

            members_scores.append(scores[: len(test_dataset) // 2])
            nonmembers_scores.append(scores[len(test_dataset) // 2 :])
            
            # Print accuracy for train and test datasets
            print(
                "train:",
                clf.score(train_dataset.data, train_dataset.label),
                "eval:",
                clf.score(test_dataset.data, test_dataset.label),
            )
            
        members_scores = torch.cat(members_scores)
        nonmembers_scores = torch.cat(nonmembers_scores)

        assert len(members_scores) == len(nonmembers_scores) == (len(test_dataset.data) + len(train_dataset.data)) // 2
        return members_scores, nonmembers_scores
    

