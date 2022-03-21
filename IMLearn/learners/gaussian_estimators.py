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
        biased_var : bool, default=False
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

        self.mu_ = np.mean(X)
        #self.mu_ = np.var(X)
        # self.mu_ = sum(X) / X.size
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
            # print("val: "+ str(X[i]) + " pdf:" + str(new_X[i]))
        return new_X

    @staticmethod
    def __pdf_of_val(val, mu, var):
        """
        Private method of calculating the pdf at one point
        """
        parameter = 1 / (var * np.sqrt(np.pi + np.pi))
        power = -0.5 * np.square((val - mu) / var)
        final_val = parameter * np.exp(power) # e^power
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
        m = X.size
        sum_xi_minus_mu = sum(X - np.zeros(shape=(m,)).fill(mu)) # Sum(i=1 to m) of (x_i - mu)^2
        parameter = m * (np.log(np.pi + np.pi) + 2*np.log(sigma))
        return -0.5 * (parameter + (sum_xi_minus_mu / np.square(sigma)))


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

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        if X.size == 0 or len(X.shape) != 2:
            raise ValueError("")

        self.mu_ = np.mean(X, axis=0)

        print(self.mu_)

        #todo: refactor?
        d = X[0].size  # col_num
        m = int(X.size / d) # num_of_data

        # #done by hand:
        # mu = np.zeros((m, 1))
        # for i in range(m):
        #     for j in range(d):
        #         mu[i] += X[i][j]
        #     mu[i] /= d
        # self.mu_ = mu

        cov_matrix = np.zeros(shape=(d, d))
        for i1 in range(d):
            for i2 in range(d):
                for k in range(m):
                    cov_matrix[i1][i2] += (X[k][i1] - self.mu_[i1]) * (X[k][i2] - self.mu_[i2])
                cov_matrix[i1][i2] /= m
        self.cov_ = cov_matrix

        self.fitted_ = True
        return self

    # @staticmethod
    # def __covariance(x1, x2, mu1, mu2, m):
    #     covariant = 0
    #     for i in range(m):
    #         covariant += (x1[i] - mu1) * (x2[i] - mu2)
    #     covariant /= m
    #     return covariant


    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
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
        m = int(X.size / X[0].size)  # num_of_data
        new_X = np.array(X)
        for i in range(m):
            new_X.put(i, self.__pdf_of_val(X[i], self.mu_, self.cov_, m))  # calculate the PDF one point and put in index i
            # print("val: " + str(X[i]) + " pdf:" + str(new_X[i]))
        return new_X

    @staticmethod
    def __pdf_of_val(val, mu, cov_matrix, num_of_data):
        """
        Private method of calculating the pdf at one point
        """
        if val.shape != mu.shape:
            print(val.shape, mu.shape)
            raise ValueError
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError
        if cov_matrix.shape[0] != val.shape[0]:
            raise ValueError
        parameter = 1/(np.sqrt(np.power(np.pi + np.pi, num_of_data) * np.linalg.det(cov_matrix)) )
        matrix_mutiplication = np.transpose(val - mu) @ np.linalg.inv(cov_matrix) @ (val - mu)
        power = -0.5 * matrix_mutiplication
        final_val = parameter * np.exp(power)  # e^power
        return final_val


    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        d = X[0].size  # col_num
        m = int(X.size / X[0].size)  # num_of_data

        sum_of_vectors = 0
        for i in range(m):
            xi_minus_mu = (X[i] - mu)  # m x 1 vector
            inside_sum = xi_minus_mu.T @ np.linalg.inv(cov) @ xi_minus_mu
            sum_of_vectors += inside_sum

        parameter = m * (d * np.log(np.pi + np.pi) + np.log(np.linalg.det(cov)))
        return -0.5 * parameter * sum_of_vectors


def univariate_gaussian_estimation(mean, variance, sample_size):
    """
    Part 3.1 Questions 1-3
    """
    # Q1
    uni = UnivariateGaussian()
    samples = np.random.normal(mean, variance, sample_size)
    uni.fit(samples)
    print("(%f, %f)" % (uni.mu_, uni.var_))

    # Q3
    sorted_samples = np.sort(samples)
    pdfs = uni.pdf(sorted_samples)
    data2 = pd.DataFrame(data={'Sorted Samples': sorted_samples, "PDF": pdfs})
    px.scatter(data2, title="PDF of the sorted samples",
               x="Sorted Samples", y="PDF", height=500).show()

    # Q2
    abs_distance = np.empty([99])
    for i in range(1, 100):
        uni.fit(samples[:10 * i:])  # fit first 10*i samples
        abs_distance[i - 1] = abs(uni.mu_ - mean)

    # todo: change the name to something more fitting
    data1 = pd.DataFrame(data={'Samples': range(10, 1000, 10), "Absolute distance": abs_distance})
    px.bar(data1, title="Sample to absolute distance between expected value and calculated expected value",
           x="Samples", y="Absolute distance", height=500).show()



def mutlivariate_gaussian_estimation(mean_vec, cov_matrix, sample_size):
    """
    Part 3.2 Questions 4-6
    """
    # Q4
    multi = MultivariateGaussian()
    samples = np.random.multivariate_normal(mean_vec, cov_matrix, sample_size)
    multi.fit(samples)
    print("mu = ", multi.mu_)
    print("cov = ", multi.cov_)

    print()

    # Q5
    AMOUNT = 20

    rows = np.linspace(-10, 10, AMOUNT)
    cols = np.linspace(-10, 10, AMOUNT)

    log_likelihood_array = np.zeros(shape=(AMOUNT, AMOUNT))

    mu1 = np.array([rows[0], 0, cols[0], 0])
    max_idx = [0,0]
    max_log_likelihood = multi.log_likelihood(mu1, multi.cov_, samples)
    for f1_i in range(AMOUNT):
        for f2_j in range(AMOUNT):
            mu1 = np.array([rows[f1_i], 0, cols[f2_j], 0])
            # print(mu1)
            log_likelihood = multi.log_likelihood(mu1, multi.cov_, samples)
            if log_likelihood > max_log_likelihood:
                max_idx[0] = f1_i
                max_idx[1] = f2_j
                max_log_likelihood = log_likelihood
            log_likelihood_array[f1_i][f2_j] =  log_likelihood

    go.Figure(go.Heatmap(x=rows, y=cols, z=log_likelihood_array),
              layout=go.Layout(title="Loglikelihood", height=600, width=600)).show()

    #Q6
    print(max_log_likelihood)



if __name__ == '__main__':
    univariate_gaussian_estimation(10, 1, 1000) # Mean = 10, Var = 1, Size = 1000 samples

    print()

    mean_vector = np.array([0, 0, 4, 0])
    covariant_matrix = np.array([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
    mutlivariate_gaussian_estimation(mean_vector, covariant_matrix, 1000)





