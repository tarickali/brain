"""
title : node.py
create : @tarickali 23/12/13
update : @tarickali 23/12/20
"""

from __future__ import annotations
from typing import Any
from brain.core.types import Array, List, Number, ArrayLike, Shape, Dtype
from brain.core.tensor import Tensor
from brain.core.tensor_utils import zeros_like, ones_like, expand_tensor, shrink_tensor

__all__ = ["Node"]

NodeLike = Tensor | ArrayLike


class Node:
    def __init__(self, data: Node | NodeLike, dtype: Dtype = float) -> None:
        data = data.data.array if isinstance(data, Node) else data
        self.data = Tensor(data, dtype=dtype)

        self.grad = zeros_like(self.data)

        self.forward = None
        self.reverse = lambda: None

        self.children = ()
        self.trainable = True

    def numpy(self) -> Array:
        return self.data.numpy()

    def item(self) -> Number:
        return self.data.item()

    def zero_grad(self) -> None:
        self.grad = zeros_like(self.grad)

    def add_children(self, nodes: tuple[Node, ...]) -> None:
        self.children += nodes

    def backward(self) -> None:
        order = list[Node]()
        visited = set[Node]()

        def build(x: Node) -> None:
            if x not in visited:
                visited.add(x)
                for child in x.children:
                    build(child)
                order.append(x)

        build(self)

        self.grad = ones_like(self.data)
        for x in reversed(order):
            x.reverse()

    # ---------------------------------------------------------#
    #################### Getter and Setter ####################
    # ---------------------------------------------------------#

    def __getitem__(self, key: int | tuple[int] | slice) -> Array | Number:
        return self.data[key]

    def __setitem__(self, key: int | tuple[int] | slice, value: ArrayLike) -> None:
        self.data[key] = value

    # ---------------------------------------------------------#
    #################### Binary Operations ####################
    # ---------------------------------------------------------#

    def __add__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)

        output = Node(data=self.data + other.data)
        output.add_children((self, other))

        def reverse():
            self.grad = expand_tensor(self.grad, output.grad.shape)
            other.grad = expand_tensor(other.grad, output.grad.shape)
            self.grad += output.grad
            other.grad += output.grad
            self.grad = shrink_tensor(self.grad, self.data.shape)
            other.grad = shrink_tensor(other.grad, other.data.shape)

        output.forward = "add"
        output.reverse = reverse

        return output

    def __sub__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)

        output = Node(data=self.data - other.data)
        output.add_children((self, other))

        def reverse():
            self.grad = expand_tensor(self.grad, output.grad.shape)
            other.grad = expand_tensor(other.grad, output.grad.shape)
            self.grad += output.grad
            other.grad += -output.grad
            self.grad = shrink_tensor(self.grad, self.data.shape)
            other.grad = shrink_tensor(other.grad, other.data.shape)

        output.forward = "sub"
        output.reverse = reverse

        return output

    def __mul__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)

        output = Node(data=self.data * other.data)
        output.add_children((self, other))

        def reverse():
            self.grad = expand_tensor(self.grad, output.grad.shape)
            other.grad = expand_tensor(other.grad, output.grad.shape)
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
            self.grad = shrink_tensor(self.grad, self.data.shape)
            other.grad = shrink_tensor(other.grad, other.data.shape)

        output.forward = "mul"
        output.reverse = reverse

        return output

    def __matmul__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)

        output = Node(data=self.data @ other.data)
        output.add_children((self, other))

        def reverse():
            self.grad = expand_tensor(self.grad, output.grad.shape)
            other.grad = expand_tensor(other.grad, output.grad.shape)
            self.grad += output.grad @ other.data.T
            other.grad += self.data.T @ output.grad
            self.grad = shrink_tensor(self.grad, self.data.shape)
            other.grad = shrink_tensor(other.grad, other.data.shape)

        output.forward = "matmul"
        output.reverse = reverse

        return output

    def __truediv__(self, other: Node | NodeLike) -> Node:
        return self * other**-1

    def __radd__(self, other: Node | Tensor | List | Number) -> Node:
        return self + other

    def __rsub__(self, other: Node | Tensor | List | Number) -> Node:
        return -self + other

    def __rmul__(self, other: Node | Tensor | List | Number) -> Node:
        return self * other

    def __rtruediv__(self, other: Node | Tensor | List | Number) -> Node:
        return self**-1 * other

    # ---------------------------------------------------------#
    ##################### Unary Operations #####################
    # ---------------------------------------------------------#

    def __pow__(self, other: Number) -> Node:
        if not isinstance(other, Number):
            raise ValueError(f"Cannot perform operation on {type(other)}")

        output = Node(data=self.data**other)
        output.add_children((self,))

        def reverse():
            grad = Tensor(other * self.data.array ** (other - 1))
            self.grad += grad * output.grad

        output.forward = "pow"
        output.reverse = reverse

        return output

    def __neg__(self) -> Node:
        output = Node(data=-self.data)
        output.add_children((self,))

        def reverse():
            self.grad += -output.grad

        output.forward = "neg"
        output.reverse = reverse

        return output

    # ---------------------------------------------------------#
    ################## Comparison Operations ##################
    # ---------------------------------------------------------#

    def __eq__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)
        return Node(data=self.data == other.data)

    def __ne__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)
        return Node(data=self.data != other.data)

    def __ge__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)
        return Node(data=self.data >= other.data)

    def __gt__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)
        return Node(data=self.data > other.data)

    def __le__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)
        return Node(data=self.data <= other.data)

    def __lt__(self, other: Node | NodeLike) -> Node:
        other = convert_node_input(other)
        return Node(data=self.data < other.data)

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"Node({self.data.array}, dtype={self.dtype}, shape={self.shape})"

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def dtype(self) -> Dtype:
        return self.data.dtype


def convert_node_input(value: Any) -> Node:
    if not isinstance(value, Node | NodeLike):
        raise ValueError(f"Cannot perform operation on {type(value)}")
    return value if isinstance(value, Node) else Node(value)
