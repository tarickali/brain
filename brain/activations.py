"""
title : activations.py
create : @tarickali 23/12/15
update : @tarickali 23/12/15
"""

from brain.core import Node, Activation
from brain.functional import *

__all__ = [
    "Identity",
    "Affine",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "ELU",
    "SELU",
    "SoftPlus",
    "Softmax",
]


class Identity(Activation):
    """Identity Activation

    Computes the elementwise function `f(x) = x`.

    """

    def func(self, x: Node) -> Node:
        return identity(x)


class Affine(Activation):
    """Affine Activation.

    Computes the function `f(x) = slope * x + intercept`.

    Parameters
    ----------
    slope : float
    intercept : float

    """

    def __init__(self, slope: float, intercept: float) -> None:
        super().__init__()
        self.slope = slope
        self.intercept = intercept

    def func(self, x: Node) -> Node:
        return affine(x, self.slope, self.intercept)


class ReLU(Activation):
    """ReLU Activation

    Computes the elementwise function `f(x) = max(x, 0)`.

    """

    def func(self, x: Node) -> Node:
        return relu(x)


class Sigmoid(Activation):
    """Sigmoid Activation

    Computes the function `f(x) = 1 / (1 + exp(-x))`.

    """

    def func(self, x: Node) -> Node:
        return sigmoid(x)


class Tanh(Activation):
    """Tanh Activation

    Computes the function `f(x) = tanh(x)`.

    """

    def func(self, x: Node) -> Node:
        return tanh(x)


class LeakyReLU(Activation):
    """LeakyReLU Activation

    Parameterized by `alpha` [float], computes the function
    ```
    f(x) = {
        x : x >= 0,
        alpha * x : x < 0
    }
    ```.

    """

    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha

    def func(self, x: Node) -> Node:
        return leaky_relu(x, self.alpha)


class ELU(Activation):
    """ELU Activation

    Parameterized by `alpha` [float], computes the function
    ```
    f(x) = {
        x : x >= 0,
        alpha * (exp(x) - 1) : x < 0
    }
    ```.

    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def func(self, x: Node) -> Node:
        return elu(x, self.alpha)


class SELU(Activation):
    """SELU Activation

    Computes the function
    ```
    f(x) = {
        SCALE * x : x >= 0,
        SCALE * ALPHA * (exp(x) - 1) : x < 0
    }
    ```
    where
    SCALE = 1.0507009873554804934193349852946,
    ALPHA = 1.6732632423543772848170429916717

    """

    def func(self, x: Node) -> Node:
        return selu(x)


class SoftPlus(Activation):
    """SoftPlus Activation

    Computes the function `f(x) = log(1 + exp(x))`.

    """

    def func(self, x: Node) -> Node:
        return softplus(x)


class Softmax(Activation):
    """Softmax Activation

    Computes the function `f(x) = exp(x) / sum(exp(x))`.

    NOTE: Returns the all ones matrix with shape of input z,
    since the categorical cross-entropy loss function L computes
    the appropriate gradient of L with respect to z.

    NOTE: The true gradient of softmax with respect to z is a Jacobian,
    and the code is given below:
    s = softmax(z)
    jacob = np.diag(s.flatten()) - np.outer(s, s)

    NOTE: It is important to note that this choice limits the use of
    the softmax activation to only the last layer of a neural network.

    """

    def func(self, x: Node) -> Node:
        return softmax(x)
