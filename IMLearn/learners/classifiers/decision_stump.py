from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> None:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_loss = np.infty
        for sign in [1, -1]:
            for j in range(X.shape[1]):
                jth_feature = X[:, j]
                threshold, thr_err = self._find_threshold(jth_feature, y, sign)
                if thr_err < min_loss:
                    min_loss = thr_err
                    self.sign_ = sign
                    self.j_ = j
                    self.threshold_ = threshold




    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        feature_to_split = X[: , self.j_]
        arr = np.array([self.sign_ if feature_to_split[i] >= self.threshold_ else -self.sign_
                         for i in range(feature_to_split.size)])
        return arr


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        lst_of_sorted_idx = np.argsort(values, axis=0)
        values, labels = values[lst_of_sorted_idx], labels[lst_of_sorted_idx]

        y_hat = np.ones(values.shape[0]) * sign
        all_err = np.ndarray(values.shape)

        thr_err = np.sum(np.abs(labels[np.sign(y_hat) != np.sign(labels)]))
        all_err[0] = thr_err
        y_hat[0] = -sign
        for i in range(1, values.shape[0]):
            if np.sign(labels[i - 1]) != sign:
                thr_err -= np.abs(labels[i - 1])
            else:
                thr_err += np.abs(labels[i - 1])
            all_err[i] = thr_err
            y_hat[i] = -sign
        return values[np.argmin(all_err)], np.min(all_err)


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_hat = self.predict(X)
        return misclassification_error(y, y_hat, normalize=True)

