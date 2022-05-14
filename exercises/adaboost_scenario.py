import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_error_lst = []
    test_error_lst = []

    adaboost = AdaBoost(lambda  : DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)

    for t in range(n_learners):
        train_err = adaboost.partial_loss(train_X, train_y, t)
        train_error_lst.append(train_err)
        test_err = adaboost.partial_loss(test_X, test_y, t)
        test_error_lst.append(test_err)

    x = np.arange(n_learners)

    fig = go.Figure(
        (go.Scatter(x=x, y=np.array(train_error_lst), fill=None, mode="lines",
                    line=dict(color="blue"),
                    name="Train Error"),
         go.Scatter(x=x, y=np.array(test_error_lst), fill=None, mode="lines",
                    line=dict(color="green"),
                    name="Test Error"))
        )
    fig.update_layout(
        title=f"Training and Test error as a function of the Number of Fitted Learners (with noise {noise})",
        xaxis_title="Number of Fitted Learners",
        yaxis_title="Loss")

    fig.show()



    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    # Returns a function with Partial Boost with iterations
    def lambda_of_predictT(iterations):
        return lambda X: adaboost.partial_predict(X, iterations)

    losses = []
    fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{iterations}}}$" for iterations in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    test_scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y,
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))
    for i, iterations in enumerate(T):
        fig.add_traces([decision_surface(lambda_of_predictT(iterations), lims[0], lims[1], showscale=False), test_scatter],
                        rows=(i // 3) + 1, cols=(i % 3) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Boundary of Predictions with Different Iterations}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()



    # Question 3: Decision surface of best performing ensemble
    for i in range(1, n_learners):
        losses.append(adaboost.partial_loss(test_X, test_y, i))
    best_T = int(np.argmin(np.array(losses)))

    lowest_test_error_graph = go.Figure([decision_surface(lambda_of_predictT(best_T), lims[0], lims[1], showscale=False),
                      test_scatter])
    lowest_test_error_graph.update_layout(title=rf"$\textbf{{Ensemble with the Lowest Test Error: {best_T} Iterations, "
                             rf"{np.min(np.array(losses))} Error}}$")
    lowest_test_error_graph.show()



    # Question 4: Decision surface with weighted samples
    new_D = adaboost.D_
    new_D = new_D / np.max(new_D) * 10
    weighted_graph = go.Figure(
        [decision_surface(lambda_of_predictT(n_learners), lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=train_y,
                                colorscale=[custom[0], custom[-1]],
                                size=new_D,
                                line=dict(color="black", width=1)))
         ])
    weighted_graph.update_layout(title=rf"$\textbf{{Training Set with Point Size Proportional to Weight (with noise {noise})}}$")
    weighted_graph.show()




if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
