from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    f = lambda x : (x+3) * (x ** 2 - 2) * (x ** 2 - 1)
    epsilon = np.random.normal(0, noise, n_samples)
    X = np.random.uniform(-1.2, 2, n_samples)
    y = f(X) + epsilon
    print("X = ", X)
    print(y)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.DataFrame(y), 1/3)
    train_X, train_y, test_X, test_y = np.ndarray.flatten(np.array(train_X)), np.array(train_y), np.array(test_X), np.array(test_y)
    #todo: understand what is "true" model
    noise_vs_noiseless_fig = go.Figure([go.Scatter(), go.Scatter(), go.Scatter()])

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    CV = 5
    lst_of_train_err= []
    lst_of_val_err = []
    for k in range(0, 11): #from 0 to 10
        poly_k = PolynomialFitting(k)

        # 0 is train error, 1 is validation error
        train_and_validation_err = cross_validate(poly_k, train_X, train_y, mean_square_error, CV)
        lst_of_train_err.append(train_and_validation_err[0])
        lst_of_val_err.append(train_and_validation_err[1])

    # todo: plot it
    train_and_validation_err_fig = go.Figure([go.Scatter(), go.Scatter(), go.Scatter()])

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(np.array(lst_of_train_err)))
    best_poly_k = PolynomialFitting(best_k)
    best_poly_k.fit(test_X, test_y)
    print(f"--------FOR CASE WITH SAMPLE SIZE {n_samples} AND NOISE {noise}---------")
    print("k* is: ", best_k)
    print("Error of polyfitted k* is: ", best_poly_k.loss(test_X, test_y))
    print()



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X,y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_X = X.sample(n=n_samples, random_state=25)
    test_X = X.drop(train_X.index)
    train_y = y.sample(fn=n_samples, random_state=25)
    test_y = y.drop(train_y.index)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    CV = 5

    lst_of_train_err_ridge = []
    lst_of_val_err_ridge = []
    lst_of_train_err_lasso = []
    lst_of_val_err_lasso = []
    for lam in range(n_evaluations):  # from 0 to 10
        ridge_lambda = RidgeRegression(lam)
        lasso_lambda = sklearn.linear_model.Lasso(alpha=lam)

        # 0 is train error, 1 is validation error
        train_and_validation_err = cross_validate(ridge_lambda, train_X, train_y, mean_square_error, CV)
        lst_of_train_err_ridge.append(train_and_validation_err[0])
        lst_of_val_err_ridge.append(train_and_validation_err[1])
        lst_of_train_err_lasso.append(train_and_validation_err[0])
        lst_of_val_err_lasso.append(train_and_validation_err[1])

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
