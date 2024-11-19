import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from scipy.stats import ttest_ind
from scipy.stats._result_classes import TtestResult
from typing import Tuple


def asserts(members_scores: np.ndarray, nonmembers_scores: np.ndarray) -> None:
    """
    Asserts for the evaluation functions.
    """
    assert len(members_scores.shape) == 1
    assert len(nonmembers_scores.shape) == 1
    assert members_scores.shape[0] > 0
    assert nonmembers_scores.shape[0] > 0
    assert members_scores.shape == nonmembers_scores.shape


def scale_and_order(
    members_scores: np.ndarray, nonmembers_scores: np.ndarray, members_lower: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale the scores and order them.
    Input: members_scores, nonmembers_scores
    Output: members_scores, nonmembers_scores
    """
    max_score = max(np.max(members_scores), np.max(nonmembers_scores))
    members_scores = members_scores / max_score
    nonmembers_scores = nonmembers_scores / max_score

    if members_lower:
        members_scores = 1 - members_scores
        nonmembers_scores = 1 - nonmembers_scores

    return members_scores, nonmembers_scores


def get_y_true_y_score(
    members_scores: np.ndarray, nonmembers_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the y_true and y_score arrays.
    Input: members_scores, nonmembers_scores
    Output: y_true, y_score
    """
    y_true = np.concatenate(
        [np.ones_like(members_scores), np.zeros_like(nonmembers_scores)]
    )
    y_score = np.concatenate([members_scores, nonmembers_scores])

    return y_true, y_score


def get_tpr_fpr(
    members_scores: np.ndarray,
    nonmembers_scores: np.ndarray,
    fpr_threshold: float,
    members_lower: bool,
) -> float:
    """
    Compute the True Positive Rate (TPR) at a given False Positive Rate (FPR) threshold.
    Input: members_scores, nonmembers_scores, fpr_threshold
    Output: TPR (float)
    """
    asserts(members_scores, nonmembers_scores)
    assert fpr_threshold >= 0 and fpr_threshold <= 1

    members_scores, nonmembers_scores = scale_and_order(
        members_scores, nonmembers_scores, members_lower
    )

    y_true, y_score = get_y_true_y_score(members_scores, nonmembers_scores)

    fpr, tpr, _ = roc_curve(y_true, y_score)

    return tpr[np.sum(fpr < fpr_threshold)]


def get_p_value(
    members_scores: np.ndarray, nonmembers_scores: np.ndarray, members_lower: bool
) -> Tuple[float, bool]:
    """
    Compute the p-value of the t-test between members_scores and nonmembers_scores.
    Input: members_scores, nonmembers_scores
    Output: p-value (float), whether members_scores < nonmembers_scores and members_scores should be lower (or > and higher) (bool)
    """
    asserts(members_scores, nonmembers_scores)

    result: TtestResult = ttest_ind(members_scores, nonmembers_scores, equal_var=False)
    p_value: float = result.pvalue
    t_statistic: float = result.statistic

    is_correct_order = (t_statistic < 0 and members_lower) or (
        t_statistic > 0 and not members_lower
    )

    return p_value, is_correct_order


def get_accuracy(
    members_scores: np.ndarray, nonmembers_scores: np.ndarray, members_lower: bool
) -> float:
    """
    Compute the accuracy of the MIA attack.
    Input: members_scores, nonmembers_scores
    Output: accuracy (float)
    """
    asserts(members_scores, nonmembers_scores)

    members_scores, nonmembers_scores = scale_and_order(
        members_scores, nonmembers_scores, members_lower
    )
    y_true, y_score = get_y_true_y_score(members_scores, nonmembers_scores)

    return accuracy_score(y_true, y_score > 0.5)


def get_auc(
    members_scores: np.ndarray, nonmembers_scores: np.ndarray, members_lower: bool
) -> float:
    """
    Compute the Area Under the Curve (AUC) of the ROC curve.
    Input: members_scores, nonmembers_scores
    Output: AUC (float)
    """
    asserts(members_scores, nonmembers_scores)

    members_scores, nonmembers_scores = scale_and_order(
        members_scores, nonmembers_scores, members_lower
    )
    y_true, y_score = get_y_true_y_score(members_scores, nonmembers_scores)

    return roc_auc_score(y_true, y_score)
