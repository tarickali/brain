"""
title : losses.py
create : @tarickali 23/12/18
update : @tarickali 23/12/18
"""

import numpy as np
from brain.core.constants import eps
from brain.core import Tensor, Node, Loss
from brain.functional import sigmoid, softmax
import brain.math as bmath

__all__ = [
    "BinaryCrossentropy",
    "CategoricalCrossentropy",
    "MeanSquaredError",
    "MeanAbsoluteError",
]


class BinaryCrossentropy(Loss):
    """BinaryCrossentropy Loss

    Computes the crossentropy loss between binary arrays y_true and y_pred
    given by: `-mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))`.

    NOTE: This `Loss` can be used when y_pred are unactivated (logits) or
    are activated.

    """

    def __init__(self, logits: bool = True) -> None:
        super().__init__()
        self.logits = logits

    def loss(self, y_true: Node, y_pred: Node) -> Node:
        assert y_true.shape == y_pred.shape
        if self.logits:
            y_pred = sigmoid(y_pred)

        pred = np.clip(y_pred.data.array, eps, 1.0 - eps)
        true = y_true.data.array

        data = -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))
        output = Node(data)
        output.add_children((y_pred,))

        def reverse():
            grad = (pred - true) / (pred * (1 - pred))
            grad = Tensor(grad / pred.size)
            y_pred.grad = grad * output.grad

        output.forward = "binary_crossentropy"
        output.reverse = reverse

        return output


class CategoricalCrossentropy(Loss):
    """CategoricalCrossentropy Loss

    Computes the crossentropy loss between multiclass arrays y_true and y_pred
    given by: `-mean(y_true * log(y_pred))`.

    NOTE: This `Loss` can be used when y_pred are unactivated (logits) or
    are activated.

    """

    def __init__(self, logits: bool = True) -> None:
        super().__init__()
        self.logits = logits

    def loss(self, y_true: Node, y_pred: Node) -> Node:
        assert y_true.shape == y_pred.shape
        if self.logits:
            y_pred = softmax(y_pred)

        pred = np.clip(y_pred.data.array, eps, 1.0 - eps)
        true = y_true.data.array

        data = -np.sum(true * np.log(pred)) / pred.shape[0]
        output = Node(data)
        output.add_children((y_pred,))

        def reverse():
            grad = Tensor((pred - true) / pred.shape[0])
            y_pred.grad = grad * output.grad

        output.forward = "categorical_crossentropy"
        output.reverse = reverse

        return output


class MeanSquaredError(Loss):
    """MeanSquaredError Loss

    Computes the squared error between y_true and y_pred given by:
    `mean((y_true - y_pred)**2)`

    """

    def loss(self, y_true: Node, y_pred: Node) -> Node:
        assert y_true.shape == y_pred.shape
        return bmath.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Loss):
    """MeanAbsoluteError Loss

    Computes the squared error between y_true and y_pred given by:
    `mean((y_true - y_pred)**2)`

    """

    def loss(self, y_true: Node, y_pred: Node) -> Node:
        assert y_true.shape == y_pred.shape
        return bmath.mean(bmath.abs(y_true - y_pred))
