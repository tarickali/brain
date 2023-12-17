"""
title : test_tensor.py
create : @tarickali 23/12/14
update : @tarickali 23/12/17
"""

import numpy as np
from brain.core.types import Array
from brain.core import Tensor


def test_init():
    # test : init with number
    array = 0
    tensor = Tensor(array)
    assert tensor.array.tolist() == array
    assert tensor.dtype == int
    assert isinstance(tensor.array, Array)

    array = 1.0
    tensor = Tensor(array)
    assert tensor.array.tolist() == array
    assert tensor.dtype == float
    assert isinstance(tensor.array, Array)

    # test : init with lists
    array = [0]
    tensor = Tensor(array)
    assert tensor.array.tolist() == array
    assert tensor.dtype == int
    assert isinstance(tensor.array, Array)

    array = [[0.0, 1.0, 2.0], [1, 2, 3]]
    tensor = Tensor(array)
    assert tensor.array.tolist() == array
    assert tensor.dtype == float
    assert isinstance(tensor.array, Array)

    # test : init with ndarray
    array = np.random.randn(2, 3)
    tensor = Tensor(array)
    assert np.all(tensor.array == array)
    assert tensor.dtype == array.dtype
    assert isinstance(tensor.array, Array)

    array = np.zeros((2, 3))
    tensor = Tensor(array, np.float32)
    assert np.all(tensor.array == array)
    assert tensor.dtype == np.float32
    assert isinstance(tensor.array, Array)


def test_binary_operations():
    #####----- test : add -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a + b
    x = Tensor(a)
    y = Tensor(b)
    z = x + y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)
    assert isinstance(z.array, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a + b
    x = Tensor(a)
    y = Tensor(b)
    z = x + y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)
    assert isinstance(z.array, Array)

    #####----- test : sub -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a - b
    x = Tensor(a)
    y = Tensor(b)
    z = x - y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)
    assert isinstance(z.array, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a - b
    x = Tensor(a)
    y = Tensor(b)
    z = x - y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)
    assert isinstance(z.array, Array)

    #####----- test : mul -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a * b
    x = Tensor(a)
    y = Tensor(b)
    z = x * y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)
    assert isinstance(z.array, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a * b
    x = Tensor(a)
    y = Tensor(b)
    z = x * y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)
    assert isinstance(z.array, Array)

    #####----- test : matmul -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 2)
    c = a @ b
    x = Tensor(a)
    y = Tensor(b)
    z = x @ y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)
    assert isinstance(z.array, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a @ b
    x = Tensor(a)
    y = Tensor(b)
    z = x @ y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)
    assert isinstance(z.array, Array)

    #####----- test : truediv -----#####
    # no broadcasting
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = a / b
    x = Tensor(a)
    y = Tensor(b)
    z = x / y
    assert isinstance(z, Tensor)
    assert np.allclose(z.array, c)
    assert isinstance(z.array, Array)

    # broadcasting
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(4)
    c = a / b
    x = Tensor(a)
    y = Tensor(b)
    z = x / y
    assert isinstance(z, Tensor)
    assert np.allclose(z.array, c)
    assert isinstance(z.array, Array)


def test_unary_operations():
    #####----- test : pow -----#####
    a = np.random.randn(2, 3)
    b = a**2
    x = Tensor(a)
    y = x**2
    assert isinstance(y, Tensor)
    assert np.all(y.array == b)
    assert isinstance(y.array, Array)

    #####----- test : neg -----#####
    a = np.random.randn(2, 3)
    b = -a
    x = Tensor(a)
    y = -x
    assert isinstance(y, Tensor)
    assert np.all(y.array == b)
    assert isinstance(y.array, Array)


def test_comparison_operations():
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    x = Tensor(a)
    y = Tensor(b)

    c = a == b
    z = x == y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)

    c = a != b
    z = x != y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)

    c = a <= b
    z = x <= y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)

    c = a < b
    z = x < y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)

    c = a >= b
    z = x >= y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)

    c = a > b
    z = x > y
    assert isinstance(z, Tensor)
    assert np.all(z.array == c)


def test_cast():
    x = Tensor([0, 1, 2])
    x.cast(int)
    assert x.dtype == int
    x.cast(np.float32)
    assert x.dtype == np.float32
