"""
title : math.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

import numpy as np
from brain.core.types import Array, Numeric
from brain.core.constants import eps
from brain.core import Node, Tensor

NodeLike = Node | Tensor | Array | Numeric

__all__ = ["sum", "mean", "exp", "log"]


def sum(x: NodeLike, axis: int | tuple[int] = None) -> Node:
    x = x if isinstance(x, Node) else Node(x)
    arr = x.data.array
    # TODO NOTE Can use the following to squeeze the array if it is a scalar:
    # `keepdims=False if axis is None or len(axis) == len(x.arr.shape) else True`
    data = np.sum(arr, axis=axis, keepdims=True)
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(np.ones_like(data))
        x.grad = grad * output.grad

    output.forward = "sum"
    output.reverse = reverse

    return output


def mean(x: NodeLike, axis: int | tuple[int] = None) -> Node:
    x = x if isinstance(x, Node) else Node(x)
    arr = x.data.array
    # TODO NOTE Can use the following to squeeze the array if it is a scalar:
    # `keepdims=False if axis is None or len(axis) == len(x.arr.shape) else True`
    data = np.mean(arr, axis=axis, keepdims=True)
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(np.full_like(data, 1.0 / (arr.size - data.size)))
        x.grad = grad * output.grad

    output.forward = "mean"
    output.reverse = reverse

    return output


def exp(x: NodeLike) -> Node:
    x = x if isinstance(x, Node) else Node(x)
    arr = x.data.array
    data = np.exp(arr)
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(data)
        x.grad = grad * output.grad

    output.forward = "exp"
    output.reverse = reverse

    return output


def log(x: NodeLike) -> Node:
    x = x if isinstance(x, Node) else Node(x)
    arr = x.data.array
    data = np.log(arr + eps)
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(1 / arr)
        x.grad = grad * output.grad

    output.forward = "log"
    output.reverse = reverse

    return output
