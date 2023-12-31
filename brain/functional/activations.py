"""
title : activations.py
create : @tarickali 23/12/20
update : @tarickali 23/12/20
"""

import numpy as np
from brain.core.types import Array, Number
from brain.core import Node, Tensor

NodeLike = Node | Tensor | Array | Number

__all__ = [
    "identity",
    "affine",
    "relu",
    "sigmoid",
    "tanh",
    "elu",
    "selu",
    "softplus",
    "softmax",
]


def identity(x: NodeLike) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    arr = x.data.array
    data = arr
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(np.ones_like(x))
        x.grad = grad * output.grad

    output.forward = "identity"
    output.reverse = reverse

    return output


def affine(x: NodeLike, slope: float = 1.0, intercept: float = 0.0) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    arr = x.data.array
    data = slope * arr + intercept
    output = Node(data)

    output.add_children((x,))

    def reverse():
        grad = Tensor(np.full_like(arr, slope))
        x.grad = grad * output.grad

    output.forward = "affine"
    output.reverse = reverse

    return output


def relu(x: NodeLike, alpha: float = 0.0) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    arr = x.data.array
    data = np.maximum(0, arr) + alpha * np.minimum(0, arr)
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(alpha + (1 - alpha) * np.heaviside(arr, 0.0, dtype=arr.dtype))
        x.grad = grad * output.grad

    output.forward = "relu"
    output.reverse = reverse

    return output


def sigmoid(x: NodeLike) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    arr = x.data.array
    data = 1 / (1 + np.exp(-arr))
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(data * (1 - data))
        x.grad = grad * output.grad

    output.forward = "sigmoid"
    output.reverse = reverse

    return output


def tanh(x: NodeLike) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    arr = x.data.array
    data = np.tanh(arr)
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(1 - data**2)
        x.grad = grad * output.grad

    output.forward = "tanh"
    output.reverse = reverse

    return output


def elu(x: NodeLike, alpha: float) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    arr = x.data.array
    data = np.where(arr >= 0, arr, alpha * (np.exp(arr) - 1))
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(np.where(arr >= 0, np.ones_like(arr), alpha * np.exp(arr)))
        x.grad = grad * output.grad

    output.forward = "elu"
    output.reverse = reverse

    return output


def selu(x: NodeLike) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    arr = x.data.array
    data = scale * (np.maximum(0, arr) + np.minimum(0, alpha * (np.exp(arr) - 1)))
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(
            np.where(arr >= 0, scale * np.ones_like(arr), alpha * scale * np.exp(arr))
        )
        x.grad = grad * output.grad

    output.forward = "selu"
    output.reverse = reverse

    return output


def softplus(x: NodeLike) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    arr = x.data.array
    e = np.exp(arr)
    data = np.log(1 + e)
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(e / (1 + e))
        x.grad = grad * output.grad

    output.forward = "softplus"
    output.reverse = reverse

    return output


def softmax(x: NodeLike, axis: int = -1) -> Node:
    x = x if isinstance(x, Node) else Node(x)

    arr = x.data.array
    e = np.exp(arr - np.max(arr))
    data = e / np.sum(e, axis=axis, keepdims=True)
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(np.ones_like(arr))
        x.grad = grad * output.grad

    output.forward = "softmax"
    output.reverse = reverse

    return output
