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
        if X.shape[0] != y.size:
            raise ValueError("Not matching idxes")
        #self.j_ = self.sign_ = self.threshold_ = 0

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

        # # By Gini values
        # list_of_gini_vals = []
        # for col in range(X.shape[1]):
        #     gini_val_for_feature = 0
        #     NUM_OF_INST = 0
        #     NUM_OF_1 = 1
        #     NUM_OF_MINUS_1 = 2
        #     #Count the number of instances
        #     num_of_samples = y.size
        #     aaaaaa = dict()
        #     for type_of_feature,i in enumerate(X[:, col]):
        #
        #
        #         if type_of_feature not in aaaaaa:
        #             # type of feature mapped to array with [num of instances, num of 1s, num of -1]
        #             aaaaaa[type_of_feature] = np.zeros(shape=3)
        #         else:
        #             aaaaaa[type_of_feature][NUM_OF_INST] += 1
        #             if y[i] == 1:
        #                 aaaaaa[type_of_feature][NUM_OF_1] += 1
        #             else:
        #                 aaaaaa[type_of_feature][NUM_OF_MINUS_1] += 1
        #     gini_vals = dict()
        #     for type_of_feature in aaaaaa.keys():
        #         arr = aaaaaa[type_of_feature]
        #
        #         type_prob = arr[NUM_OF_INST] / num_of_samples
        #         gini_val = 1 - arr[NUM_OF_1]/arr[NUM_OF_INST] - arr[NUM_OF_MINUS_1]/arr[NUM_OF_INST]
        #         gini_val_for_feature += type_prob * gini_val
        #
        #     list_of_gini_vals.append(gini_val_for_feature)
        # root_feature = np.argmin(list_of_gini_vals)
        # self.j_ = root_feature



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
        # print("feature to split is ", feature_to_split)
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
        # min_error = np.infty
        # threshold = values[0]
        #
        # for value in values:
        #     lst = np.array([sign if val>=value else -sign for val in values])
        #     indicator_if_misclassified = [1 if np.sign(lst[i]) != np.sign(labels[i]) else 0 for i in range(labels.size)]
        #
        #     error = np.dot(np.abs(labels), indicator_if_misclassified)
        #     #print(value, error)
        #     if error < min_error:
        #         min_error = error
        #         threshold = value
        #
        # #print(threshold, min_error)
        # return threshold, min_error

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


if __name__ == '__main__':
    ds = DecisionStump()
    ds1 = DecisionStump()
    ds2 = DecisionStump()

    # a = ds._find_threshold(np.array([1,2,3,4,5, 6]), np.array([1, 1, -1, 1, 1]), 1)
    #
    # ds.fit(np.array([[90, 10, 40, 30],[500, 600, 200, 300], [1,2,3,4]]).T, np.array([-1, 1, -1, 1]))
    # print(ds.j_, ds.threshold_, ds.sign_)

    ds.fit(np.array([[3], [4], [5]]), np.array([0.8, -0.2, 0.6]))
    # ds.fit(np.array([[3], [4], [5]]), np.array([0.8, -0.2, 0.6]))
    # ds.fit(np.array([[3], [4], [5]]), np.array([0.8, -0.2, 0.6]))


