from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
import pandas as pd

from IMLearn import BaseEstimator
from IMLearn.utils import split_train_test


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
    train_X, train_y, vaidate_X, validate_y = split_train_test(pd.DataFrame(X), pd.DataFrame(y), 1/cv)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    vaidate_X = np.array(vaidate_X)
    validate_y = np.array(validate_y)
    if len(train_X.shape) >= 2 and train_X.shape[1] == 1:
        train_X = np.ndarray.flatten(train_X)
        train_y = np.ndarray.flatten(train_y)
        vaidate_X = np.ndarray.flatten(vaidate_X)
        validate_y = np.ndarray.flatten(validate_y)
    train_average = cross_validate_helper(estimator, train_X, train_y, scoring, cv)
    validation_average = cross_validate_helper(estimator, vaidate_X, validate_y, scoring, cv)
    return train_average, validation_average

def cross_validate_helper(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> float:
    split_samples = np.array_split(X, cv)
    split_labels = np.array_split(y, cv)

    lst_of_losses = []
    print(np.shape(split_samples))
    for i in range(cv):
        print(np.delete(split_samples, i))
        part_X = np.concatenate(np.delete(split_samples, i, axis=0))
        part_y = np.concatenate(np.delete(split_labels, i, axis=0))
        h = estimator.fit(part_X, part_y)
        loss = scoring(part_y, h.predict(part_X))
        lst_of_losses.append(loss)
    average = np.average(np.array(lst_of_losses))
    return average



if __name__ == '__main__': #todo REMOVE. for testing
    X = np.arange(20)
    y = np.arange(20) * 10
    cv = 3
    print(X, y)

    # hi = [X[0:2], X[2:4], X[4:]]
    # print(hi)
    # for lol in hi:
    #     print(lol)

    # hi.pop()
    # print(hi)
    # split_samples = [X[i:i + 1 * int(X.shape[0] / cv)] for i in range(int(X.shape[0] / cv) - 1)].append(
    #     X[int(X.shape[0] / cv):])
    # split_labels = [y[i:i + 1 * int(y.shape[0] / cv)] for i in range(int(y.shape[0] / cv) - 1)].append(
    #     y[int(y.shape[0] / cv):])
    # print()
    split_samples = np.array_split(X, cv)
    split_labels = np.array_split(y, cv)
    print()

    print(split_samples)
    print(split_labels)
    print()
    hi = np.delete(split_samples, 2)
    print(hi)
    print()
    part_X = np.ndarray.flatten(np.concatenate(np.delete(split_samples, 1)))
    part_y = np.concatenate(np.delete(split_labels, 1))
    print(part_X)
    print(part_y)





