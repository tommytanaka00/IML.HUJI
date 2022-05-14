import numpy as np
from IMLearn.base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = y.size
        self.models_ = []
        self.weights_ = []
        self.D_ = np.array([1/m for i in range (m)])
        for t in range(self.iterations_):
            # Invote base learner
            weak_learner = self.wl_()
            weak_learner.fit(X, y * self.D_)
            self.models_.append(weak_learner)

            # Compute error
            h_X = weak_learner.predict(X)
            error = (np.sign(weak_learner.predict(X)) != np.sign(y)) @ self.D_
            # Compute weight
            if error == 0: # Should not happen
                alpha = 1
            else:
                alpha = 1/2 * np.log(1/error - 1)
            self.weights_.append(alpha)

            # Update sample weight
            self.D_ *= np.exp(-alpha * y * h_X)
            # Normalize
            sum_of_D = sum(self.D_)
            self.D_ /= sum_of_D



    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

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
        return misclassification_error(y, y_hat)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if T > self.iterations_:
            T = self.iterations_

        responses = self.weights_[0] * self.models_[0].predict(X)
        for t in range(1, T):
            responses += self.weights_[t] * self.models_[t].predict(X)
        return np.sign(responses)


    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_hat = self.partial_predict(X, T)
        return misclassification_error(y, y_hat)

