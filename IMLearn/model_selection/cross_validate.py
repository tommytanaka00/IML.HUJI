from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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
    # if len(X.shape) >= 2 and X.shape[1] == 1:
    #     X = np.ndarray.flatten(X)
    #     y = np.ndarray.flatten(y)
    # # p = np.random.permutation(len(y))
    # # X,y = X[p], y[p]
    #
    # X, y = shuffle(X, y, random_state=0)
    split_samples = np.array_split(X, cv)
    split_labels = np.array_split(y, cv)

    lst_of_losses_train = []
    lst_of_losses_val = []
    # print(np.shape(split_samples))
    for i in range(cv):
        # print(np.delete(split_samples, i))
        train_X = np.concatenate(np.delete(split_samples, i, axis=0))
        validate_X = split_samples[i]
        train_y = np.concatenate(np.delete(split_labels, i, axis=0))
        validate_y = split_labels[i]
        h = estimator.fit(train_X, train_y)

        loss = scoring(train_y, h.predict(train_X))
        loss2 = scoring(validate_y, h.predict(validate_X))
        lst_of_losses_train.append(loss)
        lst_of_losses_val.append(loss2)
    train_average = np.average(np.array(lst_of_losses_train))
    validation_average = np.average(np.array(lst_of_losses_val))
    return train_average, validation_average