"""
title : test_node.py
create : @tarickali 23/12/14
update : @tarickali 23/12/17
"""

import numpy as np
from brain.core import Node, Tensor


def test_init():
    # test : init with number
    data = 0
    node = Node(data)
    assert node.data.array.tolist() == data
    print(node.dtype)
    assert node.dtype == int
    assert isinstance(node.data, Tensor)
    assert isinstance(node.grad, Tensor)

    data = 1.0
    node = Node(data)
    assert node.data.array.tolist() == data
    assert node.dtype == float
    assert isinstance(node.data, Tensor)
    assert isinstance(node.grad, Tensor)

    # test : init with lists
    data = [0]
    node = Node(data)
    assert node.data.array.tolist() == data
    assert node.dtype == int
    assert isinstance(node.data, Tensor)
    assert isinstance(node.grad, Tensor)

    data = [[0.0, 1.0, 2.0], [1, 2, 3]]
    node = Node(data)
    assert node.data.array.tolist() == data
    assert node.dtype == float
    assert isinstance(node.data, Tensor)
    assert isinstance(node.grad, Tensor)

    # test : init with ndarray
    data = np.random.randn(2, 3)
    node = Node(data)
    assert np.all(node.data.array == data)
    assert node.dtype == data.dtype
    assert isinstance(node.data, Tensor)
    assert isinstance(node.grad, Tensor)

    data = np.zeros((2, 3))
    node = Node(data, np.float32)
    assert np.all(node.data.array == data)
    assert node.dtype == np.float32
    assert isinstance(node.data, Tensor)
    assert isinstance(node.grad, Tensor)


def test_binary_operations():
    #####----- test : add -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a + b
    x = Node(a)
    y = Node(b)
    z = x + y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a + b
    x = Node(a)
    y = Node(b)
    z = x + y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    #####----- test : sub -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a - b
    x = Node(a)
    y = Node(b)
    z = x - y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a - b
    x = Node(a)
    y = Node(b)
    z = x - y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    #####----- test : mul -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a * b
    x = Node(a)
    y = Node(b)
    z = x * y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a * b
    x = Node(a)
    y = Node(b)
    z = x * y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    #####----- test : matmul -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 2)
    c = a @ b
    x = Node(a)
    y = Node(b)
    z = x @ y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a @ b
    x = Node(a)
    y = Node(b)
    z = x @ y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    #####----- test : truediv -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a / b
    x = Node(a)
    y = Node(b)
    z = x / y
    assert isinstance(z, Node)
    assert np.allclose(z.data.array, c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a / b
    x = Node(a)
    y = Node(b)
    z = x / y
    assert isinstance(z, Node)
    assert np.allclose(z.data.array, c)
    assert isinstance(z.data, Tensor)
    assert isinstance(z.grad, Tensor)


def test_unary_operations():
    #####----- test : pow -----#####
    a = np.random.randn(2, 3)
    b = a**2
    x = Node(a)
    y = x**2
    assert isinstance(y, Node)
    assert np.all(y.data.array == b)
    assert isinstance(y.data, Tensor)
    assert isinstance(y.grad, Tensor)

    #####----- test : neg -----#####
    a = np.random.randn(2, 3)
    b = -a
    x = Node(a)
    y = -x
    assert isinstance(y, Node)
    assert np.all(y.data.array == b)
    assert isinstance(y.data, Tensor)
    assert isinstance(y.grad, Tensor)


def test_comparison_operations():
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    x = Node(a)
    y = Node(b)

    c = a == b
    z = x == y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)

    c = a != b
    z = x != y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)

    c = a <= b
    z = x <= y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)

    c = a < b
    z = x < y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)

    c = a >= b
    z = x >= y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)

    c = a > b
    z = x > y
    assert isinstance(z, Node)
    assert np.all(z.data.array == c)


def test_backward():
    pass
