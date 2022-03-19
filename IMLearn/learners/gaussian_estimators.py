from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet

import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        #raise NotImplementedError()
        self.mu_ = sum(X) / X.size
        self.var_ = (sum(np.multiply(X, X)) / X.size) - (np.square(self.mu_))


        # var2 = (1/(X.size-1)) * sum(X2)
        # print("var1 = %f, var2 = %f", self.var_, var2)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        new_X = np.array(X)
        for i in range(X.size):
            new_X.put(i, self.__pdf_of_val(X[i], self.mu_, self.var_)) # calculate the PDF one point and put in index i
            print("val: "+ str(X[i]) + " pdf:" + str(new_X[i]))
        return new_X

    @staticmethod
    def __pdf_of_val(val, mu, var):
        """
        Private method of calculating the pdf at one point
        """
        parameter = 1 / (var * np.sqrt(np.pi + np.pi))
        power = -0.5 * np.square((val - mu) / var)
        final_val = parameter * np.power(np.e, power) # e^power
        return final_val




    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        raise NotImplementedError()


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        #todo: refactor?
        d = X[0].size  # col_num
        m = int(X.size / d) # num_of_data
        mu = np.zeros((m, 1))

        for i in range(m):
            for j in range(d):
                mu[i] += X[i][j]
            mu[i] /= d
        self.mu_ = mu

        cov_matrix = np.zeros(shape=(m, m))
        for i1 in range(m):
            for i2 in range(m):
                cov_matrix[i1][i2] = self.__covariance(X[i1], X[i2], self.mu_[i1], self.mu_[i2], d)
        self.cov_ = cov_matrix

        print(cov_matrix)

        self.fitted_ = True
        return self

    @staticmethod
    def __covariance(x1, x2, mu1, mu2, d):
        covariant = 0
        for i in range(d):
            covariant += (x1[i] - mu1) * (x2[i] - mu2)
        covariant /= d
        return covariant


    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        raise NotImplementedError()

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        raise NotImplementedError()


if __name__ == '__main__':
    # MEAN = 10
    # VARIANCE = 1
    # SAMPLE_SIZE = 1000
    #
    # uni = UnivariateGaussian()
    # thousand_samples_uni = np.random.normal(MEAN,  VARIANCE, 1000) # Mean = 10, Var = 1, Size = 1000 samples
    #
    # #fit a thousand gaussian samples
    # uni.fit(thousand_samples_uni)
    #
    # print("(%f, %f)" % (uni.mu_, uni.var_)) # Q1
    #
    #
    # sorted_samples = np.sort(thousand_samples_uni)
    # pdfs = uni.pdf(sorted_samples)
    # data2 = pd.DataFrame(data={'Sorted Samples': sorted_samples, "PDF": pdfs})
    # px.scatter(data2, title="PDF of the sorted samples",
    #        x="Sorted Samples", y="PDF", height=500).show()  # Q2
    #
    #
    # means = np.empty([99]) # todo: do we need this?
    # abs_distance = np.empty([99])
    # for i in range(1, 100):
    #     uni.fit(thousand_samples_uni[:10*i:]) # fit first 10*i samples
    #     means[i - 1] = uni.mu_
    #     abs_distance[i - 1] = abs(uni.mu_ - MEAN)
    # #abs_distance = np.array([abs(i - MEAN) for i in means])
    #
    # #todo: change the name to something more fitting
    # data1 = pd.DataFrame(data={'Samples': range(10, 1000, 10), "Absolute distance": abs_distance})
    # px.bar(data1, title="Sample to absolute distance between expected value and calculated expected value",
    #        x="Samples", y="Absolute distance", height=500).show()

    MEAN_MULTI = np.array([0, 0, 4, 0])
    VARIANCE_MULTI = np.array([[1, 0.2, 0, 0.5],
                                 [0.2, 2, 0, 0],
                                 [0,   0, 1, 0],
                                 [0.5, 0, 0, 1]])
    SAMPLE_SIZE = 10

    multi = MultivariateGaussian()
    thousand_samples_multi = np.random.multivariate_normal(MEAN_MULTI, VARIANCE_MULTI, SAMPLE_SIZE)
    # multi.fit(np.array([[2,4,6,8,10],
    #                     [7,3,5,1,9]]))

    #Q1
    multi.fit(thousand_samples_multi)
    print(multi.mu_)
    #sa
    print(multi.cov_)




