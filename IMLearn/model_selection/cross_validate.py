from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    split_samples = np.array_split(X, cv)
    split_labels = np.array_split(y, cv)
    lst_of_losses = []
    for i in range(cv):
        part_X = np.concatenate(np.delete(split_samples, 1))
        part_y = np.concatenate(np.delete(split_labels, 1))
        h = estimator.fit(part_X, part_y)
        lst_of_losses.append(h.loss(part_X, part_y))
    average = np.average(np.array(lst_of_losses))
    return average, 0



if __name__ == '__main__': #todo REMOVE. for testing
    X = np.arange(20).reshape(5, 4)
    y = np.arange(5) * 10
    cv = 3
    print(X, y)
    split_samples = np.array_split(X, cv)
    split_labels = np.array_split(y, cv)
    print()

    print(split_samples)
    print(split_labels)
    print()

    part_X = np.ndarray.flatten(np.concatenate(np.delete(split_samples, 1)))
    part_y = np.concatenate(np.delete(split_labels, 1))
    print(part_X)
    print(part_y)



