from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets as datasets
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
    f = lambda x : (x+3) * (x ** 2 - 4) * (x ** 2 - 1)
    epsilon = np.random.normal(0, noise, n_samples)
    X = np.random.uniform(-1.2, 2, n_samples)
    noise_less_y = np.array(f(X))
    y = np.array(f(X) + epsilon)

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.DataFrame(y), 2/3)
    train_X = np.ndarray.flatten(np.array(train_X))
    train_y = np.ndarray.flatten(np.array(train_y))
    test_X = np.ndarray.flatten(np.array(test_X))
    test_y = np.ndarray.flatten(np.array(test_y))
    noise_vs_noiseless_fig = go.Figure( [
        go.Scatter(x=X, y=noise_less_y, fill=None, mode="markers",
                    name="Noiseless y"),
        go.Scatter(x=train_X, y=train_y, fill=None, mode="markers",
                    name="Train y"),
        go.Scatter(x=test_X, y=test_y, fill=None, mode="markers",
                   name="Test y")
    ])
    noise_vs_noiseless_fig.update_layout(
        title=f"Scatter plot of {n_samples} samples of f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) with noise {noise} and without",
        xaxis_title="x",
        yaxis_title="f(x)")
    noise_vs_noiseless_fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    CV = 5
    lst_of_train_err= []
    lst_of_val_err = []
    from_0_to_10 = np.array([i for i in range(0, 11)])
    for k in from_0_to_10:
        poly_k = PolynomialFitting(k)

        # 0 is train error, 1 is validation error
        train_and_validation_err = cross_validate(poly_k, train_X, train_y, mean_square_error, CV)
        lst_of_train_err.append(train_and_validation_err[0])
        lst_of_val_err.append(train_and_validation_err[1])

    # todo: plot it
    train_and_validation_err_fig = go.Figure(
        [go.Scatter(x=from_0_to_10, y=np.array(lst_of_train_err), mode="lines+markers",
                    line=dict(color="blue"),
                    name="Train Error"),
         go.Scatter(x=from_0_to_10, y=np.array(lst_of_val_err), mode="lines+markers",
                    line=dict(color="green"),
                    name="Validation Error")]
        )
    train_and_validation_err_fig.update_layout(
        title=f"Training and Validation Error of different degrees of Polynomial Fit"
              f" (with noise {noise})",
        xaxis_title="Polynomial fit degree",
        yaxis_title="Loss")
    train_and_validation_err_fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(np.array(lst_of_val_err)))
    best_poly_k = PolynomialFitting(best_k)
    best_poly_k.fit(train_X, train_y)
    best_poly_k_loss = round(best_poly_k.loss(test_X, test_y), 2)
    validation_error = round(np.min(np.array(lst_of_val_err)), 2)
    print(f"--------FOR CASE WITH SAMPLE SIZE {n_samples} AND NOISE {noise}---------")
    print("k* is: ", best_k)
    print(f"Test error of polyfitted k* is {best_poly_k_loss}, while validation error was {validation_error}", )
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
    X,y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    #print(np.shape(X), np.shape(y))
    train_X = X.sample(n=n_samples, random_state=25)
    test_X = X.drop(train_X.index)
    train_y = y.sample(n=n_samples, random_state=25)
    test_y = y.drop(train_y.index)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    CV = 5

    lst_of_train_err_ridge = []
    lst_of_val_err_ridge = []
    lst_of_train_err_lasso = []
    lst_of_val_err_lasso = []

    x_range = np.linspace(0.001, 1, num=n_evaluations)
    for lam in x_range:
        ridge_lambda = RidgeRegression(lam, include_intercept=True)
        lasso_lambda = Lasso(alpha=lam)

        # 0 is train error, 1 is validation error
        train_and_val_err_ridge = cross_validate(ridge_lambda, train_X, train_y, mean_square_error, CV)
        lst_of_train_err_ridge.append(train_and_val_err_ridge[0])
        lst_of_val_err_ridge.append(train_and_val_err_ridge[1])

        train_and_val_err_lasso = cross_validate(lasso_lambda, train_X, train_y, mean_square_error, CV)
        lst_of_train_err_lasso.append(train_and_val_err_lasso[0])
        lst_of_val_err_lasso.append(train_and_val_err_lasso[1])

    #fig = make_subplots(rows=1, cols=2, start_cell="bottom-left")
    fig = go.Figure(
        [go.Scatter(x=x_range, y=lst_of_train_err_ridge, mode="lines",
                    line=dict(color="blue"), name="Ridge Train Error"),
         go.Scatter(x=x_range, y=lst_of_val_err_ridge, mode="lines",
                    line=dict(color="red"), name="Ridge Validation Error"),
         go.Scatter(x=x_range, y=lst_of_train_err_lasso, mode="lines",
                     line=dict(color="green"), name="Lasso Train Error"),
          go.Scatter(x=x_range, y=lst_of_val_err_lasso, mode="lines",
                     line=dict(color="orange"), name="Lasso Validation Error")
          ]
        )
    fig.update_layout(
        title=f"Training and Validation Error of Ridge and Lasso Regularization ",
        xaxis_title="Regularization Parameters",
        yaxis_title="Loss")
    fig.show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lambda_ridge = x_range[int(np.argmin(np.array(lst_of_val_err_ridge)))]
    ridge_lambda = RidgeRegression(best_lambda_ridge, include_intercept=True)
    ridge_lambda.fit(train_X, train_y)
    ridge_loss = ridge_lambda.loss(test_X, test_y)

    best_lambda_lasso = x_range[int(np.argmin(np.array(lst_of_val_err_lasso)))]
    lasso_lambda = Lasso(best_lambda_lasso)
    lasso_lambda.fit(train_X, train_y)
    lasso_loss = mean_square_error(test_y, lasso_lambda.predict(test_X))

    linear_reg = LinearRegression(include_intercept=True)
    linear_reg.fit(train_X, train_y)
    linear_loss = linear_reg.loss(test_X, test_y)

    print(f"--------FOR CASE WITH {n_samples} SAMPLES FOR TRAINING AND {n_evaluations} EVALUATIONS---------")
    print(f"Best ridge regularization parameter: {best_lambda_ridge}")
    print(f"Best lasso regularization parameter: {best_lambda_lasso}")

    print(f"Ridge Error: {ridge_loss}")
    print(f"Lasso Error: {lasso_loss}")
    print(f"Least Squares Error: {linear_loss}")



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
