"""
title : utils.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

import numpy as np
from brain.core import Tensor, Node

NodeLike = Node | Tensor

__all__ = ["flatten"]


def flatten(x: NodeLike) -> Node:
    x = x if isinstance(x, Node) else Node(x)
    arr = x.data.array
    data = arr.reshape(-1, np.prod(arr.shape[1:]))
    output = Node(data)
    output.add_children((x,))

    def reverse():
        grad = Tensor(output.grad.array.reshape(arr.shape))
        x.grad += grad

    output.forward = "flatten"
    output.reverse = reverse

    return output
