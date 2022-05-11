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
    train_error = np.zeros(shape=n_learners)
    test_error = np.zeros(shape=n_learners)

    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)
    lol = adaboost.loss(train_X, train_y)
    lolol = adaboost.loss(test_X, test_y)

    print(lol)
    print(lolol)
    for i in range(n_learners):
        adaboost = AdaBoost(lambda :DecisionStump(), i+1)
        adaboost.fit(train_X, train_y)
        train_error[i] = adaboost.loss(train_X, train_y)
        test_error[i] = adaboost.loss(test_X, test_y)


    go.Figure(data=[go.Line(x=np.linspace(0, n_learners, 1), y=[train_error, test_error], mode="markers", showlegend=False
                               # ,marker=dict(color=y, symbol=symbols[y], line=dict(color="black", width=1),
                               #             colorscale=[custom[0], custom[-1]])
                            )],
              layout=go.Layout(title=rf"$\textbf{{(1) LOL Dataset}}$")).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{val}}}$" for val in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    raise NotImplementedError


    for i,val in enumerate(T):
        print(val, i)
        # adaboost.partial_predict(test_X, val)
        fig.add_traces([decision_surface(adaboost.fit(train_X,train_y).predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_X,
                                               #$symbol=symbols[y],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                        rows=(i // 3) + 1, cols=(i % 3) + 1)


    fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models - Dataset}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 10, 20, 5)
