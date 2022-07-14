import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return (1/y_true.size) * sum(np.subtract(y_true, y_pred) ** 2)


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    misclassification_amount = 0
    for i in range(y_true.size):
        if y_true[i] != y_pred[i]:
            misclassification_amount += 1
    if normalize:
        misclassification_amount /= y_true.size
    return misclassification_amount

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    return 1 - misclassification_error(y_true, y_pred, True)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)

    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    raise NotImplementedError()