"""
title : initializers.py
create : @tarickali 23/12/15
update : @tarickali 23/12/15
"""


import numpy as np
from brain.core.types import Shape
from brain.core import Tensor, Initializer

__all__ = [
    "Zeros",
    "Ones",
    "Constant",
    "RandomUniform",
    "RandomNormal",
    "XavierUniform",
    "XavierNormal",
    "HeUniform",
    "HeNormal",
    "LecunUniform",
    "LecunNormal",
]


class Zeros(Initializer):
    """Zeros Initializer

    Initializes a Tensor with zeros.

    """

    def init(self, shape: Shape = None) -> Tensor:
        return Tensor(np.zeros(shape))


class Ones(Initializer):
    """Ones Initializer

    Initializes a Tensor with ones.

    """

    def init(self, shape: Shape = None) -> Tensor:
        return Tensor(np.ones(shape))


class Constant(Initializer):
    """Constant Initializer

    Initializes a Tensor with a constant value.

    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def init(self, shape: Shape = None) -> Tensor:
        return Tensor(np.full(shape, self.value))


class RandomUniform(Initializer):
    """RandomUniform Initializer

    Initializes a Tensor with values sampled from `U(low, high)` where
    `low` and `high` are the given low and high range values for the
    distribution, respectively.

    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def init(self, shape: Shape = None) -> Tensor:
        return Tensor(np.random.uniform(self.low, self.high, shape))


class RandomNormal(Initializer):
    """RandomNormal Initializer

    Initializes a Tensor with values sampled from `N(mu, std)` where
    `mu` and `std` are the given mean and standard deviation, respectively.

    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def init(self, shape: Shape = None) -> Tensor:
        return Tensor(np.random.normal(self.mean, self.std, shape))


class XavierUniform(Initializer):
    """XavierUniform Initializer

    Initializes a Tensor with values sampled from `U(-limit, limit)` with
    `limit = sqrt(6.0 / (fan_in + fan_out))` where `fan_in` and `fan_out`
    are the number of input and output units to the Tensor, respectively.

    """

    def init(self, shape: Shape = None) -> Tensor:
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return Tensor(np.random.uniform(-limit, limit, shape))


class XavierNormal(Initializer):
    """XavierNormal Initializer

    Initializes a Tensor with values sampled from `N(0.0, std)` with
    `std = sqrt(2.0 / (fan_in + fan_out))` where `fan_in` and `fan_out`
    are the number of input and output units to the Tensor, respectively.

    """

    def init(self, shape: Shape = None) -> Tensor:
        std = np.sqrt(2.0 / (shape[0] + shape[1]))
        return Tensor(np.random.normal(0.0, std, shape))


class HeUniform(Initializer):
    """HeUniform Initalizer

    Initializes a Tensor with values sampled from `U(-limit, limit)` with
    `limit = sqrt(6.0 / fan_in)` where `fan_in` is the number of input units
    to the Tensor.

    """

    def init(self, shape: Shape = None) -> Tensor:
        limit = np.sqrt(6.0 / shape[0])
        return Tensor(np.random.uniform(-limit, limit, shape))


class HeNormal(Initializer):
    """HeNormal Initalizer

    Initializes a Tensor with values sampled from `N(0.0, std)` with
    `std = sqrt(2.0 / fan_in)` where `fan_in` is the number of input units
    to the Tensor.

    """

    def init(self, shape: Shape = None) -> Tensor:
        std = np.sqrt(2.0 / shape[0])
        return Tensor(np.random.normal(0.0, std, shape))


class LecunUniform(Initializer):
    """LecunUniform Initializer

    Initializes a Tensor with values sampled from `U(-limit, limit)` with
    `limit = sqrt(3.0 / fan_in)` where `fan_in` is the number of input units
    to the Tensor.

    """

    def init(self, shape: Shape = None) -> Tensor:
        limit = np.sqrt(3.0 / shape[0])
        return Tensor(np.random.uniform(-limit, limit, shape))


class LecunNormal(Initializer):
    """LecunNormal Initializer

    Initializes a Tensor with values sampled from `N(0.0, std)` with
    `std = sqrt(1.0 / fan_in)` where `fan_in` is the number of input units
    to the Tensor.

    """

    def init(self, shape: Shape = None) -> Tensor:
        std = np.sqrt(1.0 / shape[0])
        return Tensor(np.random.normal(0.0, std, shape))
