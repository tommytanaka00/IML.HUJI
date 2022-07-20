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
            Size of input to layer (number of neurons in preceding layer)

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
        # if not activation:
        #     self.activation_ = None # todo: change to Linear (how?)
        # else:
        #     self.activation_ = activation
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
        assert X.shape[1] == self.input_dim_
        if self.include_intercept_:
            X = np.insert(X, 0, values=1, axis=1)

        output = X @ self.weights_
        if self.activation_ is not None:
            output = self.activation_.compute_output(X=output, y=y)

        assert output.shape[1] == self.output_dim_
        return output #todo: check

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
        return X.T


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
        # y is the class targets


        cross_entropy_loss_arr = []
        softmax_X = softmax(X)
        for i,x in enumerate(np.array([[0.7, 0.1, 0.2]])):
            rename_this_list = []
            e_k = np.zeros(shape=x.shape)
            e_k[y[i]] = 1
            cross_entropy_loss_arr.append(cross_entropy(e_k, x))
            # sum_of_losses = sum(rename_this_list)

            # cross_entropy_loss_arr.append(sum_of_losses)
        assert len(cross_entropy_loss_arr) == X.shape[0]
        return np.array(cross_entropy_loss_arr)


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
        # for x in X:




# todo for testing, remove!
if __name__ == '__main__':
    softmax_output = np.array([[0.7, 0.1, 0.2]])
    print(softmax(softmax_output))

    X = np.array([[0.7, 0.1, 0.2],
         [0.1, 0.5, 0.4],
         [0.02, 0.9, 0.08]])
    y = np.array([0])
    print(CrossEntropyLoss().compute_output(X=softmax_output, y=y))
    # print(CrossEntropyLoss().compute_output(X=X, y=y))

    print("\n\n\n")
    # X = np.array([[1,4,2], [1,2,2]])
    # relu_matrix = relu.compute_output(X)
    # print(relu_matrix)
    # print(X)

    # print(softmax(X))
    # e_1 = np.zeros(shape=X.shape[1])
    # e_1[0] = 1
    # e_2 = np.zeros(shape=X.shape[1])
    # e_2[1] = 1
    # e_3 = np.zeros(shape=X.shape[1])
    # e_3[2] = 1
    # e = [e_1, e_2, e_3]
    # print(e_1, e_2, e_3)
    # print(cross_entropy(e_1, softmax(X)[0]))
    # print(cross_entropy(e_2, softmax(X)[0]))
    # print(cross_entropy(e_3, softmax(X)[0]))
    # y = [cross_entropy(e_i, softmax(X)[0]) for e_i in e]

    # print(end="\n\n")
    # print(CrossEntropyLoss().compute_output(X, np.array([0,0,1])))

    relu = ReLU()

    X = np.array([[1, 2, 3, 2.5]])
    y = [1]  # y is class target

    layer = FullyConnectedLayer(4, 3, relu, False)
    #print("weights= ", layer.weights_, end="\n\n")

    output_of_layer1 = layer.compute_output(X=X)
    print(output_of_layer1)

    layer2 = FullyConnectedLayer(3, 2, CrossEntropyLoss(), False)
    print("\n")
    #print("\nweights= ", layer2.weights_, end="\n\n")

    print(layer2.compute_output(X=output_of_layer1, y=y))