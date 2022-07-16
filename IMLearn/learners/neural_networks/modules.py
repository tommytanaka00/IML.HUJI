import numpy as np
from IMLearn.base.base_module import BaseModule
from IMLearn.metrics.loss_functions import cross_entropy, softmax


class FullyConnectedLayer(BaseModule):
    """
    Module of a fully connected layer in a neural network

    Attributes:
    -----------
    input_dim_: int
        Size of input to layer (number of neurons in preceding layer

    output_dim_: int
        Size of layer output (number of neurons in layer_)

    activation_: BaseModule
        Activation function to be performed after integration of inputs and weights

    weights: ndarray of shape (input_dim_, outout_dim_)
        Parameters of function with respect to which the function is optimized.

    include_intercept: bool
        Should layer include an intercept or not
    """
    def __init__(self, input_dim: int, output_dim: int, activation: BaseModule = None, include_intercept: bool = True):
        """
        Initialize a module of a fully connected layer

        Parameters:
        -----------
        input_dim: int
            Size of input to layer (number of neurons in preceding layer

        output_dim: int
            Size of layer output (number of neurons in layer_)

        activation_: BaseModule, default=None
            Activation function to be performed after integration of inputs and weights. If
            none is specified functions as a linear layer

        include_intercept: bool, default=True
            Should layer include an intercept or not

        Notes:
        ------
        Weights are randomly initialized following N(0, 1/input_dim)
        """
        super().__init__()
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim
        self.include_intercept_ = include_intercept
        if not activation:
            self.activation_ = None # todo: change to Linear (how?)
        else:
            self.activation_ = activation
        self.weights_ = np.random.normal(0, 1/input_dim, size=(input_dim, output_dim))

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute activation(weights @ x) for every sample x: output value of layer at point
        self.weights and given input

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        --------
        output: ndarray of shape (n_samples, output_dim)
            Value of function at point self.weights
        """
        # todo: understand what to do with the kwargs

        if self.include_intercept_:
            X = np.insert(X, 0, values=1, axis=1)

        # output = []
        # for sample in X:
        #     output.append(self.weights_ @ sample)
        # print(np.array(output))

        return X @ self.weights_  #todo: I guess we don't do sigmoid?

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point self.weights

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        -------
        output: ndarray of shape (input_dim, n_samples)
            Derivative with respect to self.weights at point self.weights
        """
        raise NotImplementedError()


class ReLU(BaseModule):
    """
    Module of a ReLU activation function computing the element-wise function ReLU(x)=max(x,0)
    """

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute element-wise value of activation

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be passed through activation

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            Data after performing the ReLU activation function
        """
        # todo: understand what to do with the kwargs
        return np.maximum(X, np.zeros(shape=X.shape))

    def compute_jacobian(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to given data

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to compute derivative with respect to

        Returns:
        -------
        output: ndarray of shape (n_samples,)
            Element-wise derivative of ReLU with respect to given data
        """
        raise NotImplementedError()


class CrossEntropyLoss(BaseModule):
    """
    Module of Cross-Entropy Loss: The Cross-Entropy between the Softmax of a sample x and e_k for a true class k
    """
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the Cross-Entropy over the Softmax of given data, with respect to every

        CrossEntropy(Softmax(x),e_k) for every sample x

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data for which to compute the cross entropy loss

        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed

        Returns:
        --------
        output: ndarray of shape (n_samples,)
            cross-entropy loss value of given X and y
        """
        raise NotImplementedError()

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the derivative of the cross-entropy loss function with respect to every given sample

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data with respect to which to compute derivative of the cross entropy loss

        y: ndarray of shape (n_samples,)
            Values with respect to which cross-entropy loss is computed

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            derivative of cross-entropy loss with respect to given input
        """
        raise NotImplementedError()


# todo for testing, remove!
if __name__ == '__main__':
    relu = ReLU()
    X = np.array([[1,4,2], [-4, 4, -2], [-10, -2, 3], [2, -4, 100]])
    relu_matrix = relu.compute_output(X)
    print(relu_matrix)
    print(X)