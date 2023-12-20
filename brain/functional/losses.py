"""
title : losses.py
create : @tarickali 23/12/20
update : @tarickali 23/12/20
"""

import numpy as np
from brain.core.constants import eps
from brain.core import Tensor, Node
from brain.functional import sigmoid, softmax
import brain.math as bmath

__all__ = [
    "binary_crossentropy",
    "categorical_crossentropy",
    "mean_squared_error",
    "mean_absolute_error",
]


def binary_crossentropy(true: Node, pred: Node, logits: bool = True) -> Node:
    assert true.shape == pred.shape
    if logits:
        pred = sigmoid(pred)

    pred_arr = np.clip(pred.data.array, eps, 1.0 - eps)
    true_arr = true.data.array

    data = -np.mean(true_arr * np.log(pred_arr) + (1 - true_arr) * np.log(1 - pred_arr))
    output = Node(data)
    output.add_children((pred,))

    def reverse():
        grad = (pred_arr - true_arr) / (pred_arr * (1 - pred_arr))
        grad = Tensor(grad / pred_arr.size)
        pred.grad = grad * output.grad

    output.forward = "binary_crossentropy"
    output.reverse = reverse

    return output


def categorical_crossentropy(true: Node, pred: Node, logits: bool = True) -> Node:
    assert true.shape == pred.shape
    if logits:
        pred = softmax(pred)

    pred_arr = np.clip(pred.data.array, eps, 1.0 - eps)
    true_arr = true.data.array

    data = -np.sum(true_arr * np.log(pred_arr)) / pred_arr.shape[0]
    output = Node(data)
    output.add_children((pred,))

    def reverse():
        grad = Tensor((pred_arr - true_arr) / pred_arr.shape[0])
        pred.grad = grad * output.grad

    output.forward = "categorical_crossentropy"
    output.reverse = reverse

    return output


def mean_squared_error(true: Node, pred: Node) -> Node:
    assert true.shape == pred.shape
    return bmath.mean((true - pred) ** 2)


def mean_absolute_error(true: Node, pred: Node) -> Node:
    assert true.shape == pred.shape
    return bmath.mean(bmath.abs(true - pred))
