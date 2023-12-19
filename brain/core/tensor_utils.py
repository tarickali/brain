"""
title : utils.py
create : @tarickali 23/12/13
update : @tarickali 23/12/18
"""

import numpy as np
from brain.core.types import Array, Shape, Dtype
from brain.core.tensor import Tensor, Numeric

__all__ = [
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full_like",
    "expand_tensor",
    "shrink_tensor",
]


def zeros(shape: Shape, dtype: Dtype) -> Tensor:
    return Tensor(np.zeros(shape=shape, dtype=dtype))


def zeros_like(tensor: Tensor | Array) -> Tensor:
    array = tensor.array if isinstance(tensor, Tensor) else tensor
    return Tensor(np.zeros_like(a=array))


def ones(shape: Shape, dtype: Dtype) -> Tensor:
    return Tensor(np.ones(shape=shape, dtype=dtype))


def ones_like(tensor: Tensor | Array) -> Tensor:
    array = tensor.array if isinstance(tensor, Tensor) else tensor
    return Tensor(np.ones_like(a=array))


def full(shape: Shape, value: Numeric) -> Tensor:
    return Tensor(np.full(shape=shape, fill_value=value))


def full_like(tensor: Tensor | Array, value: Numeric) -> Tensor:
    array = tensor.array if isinstance(tensor, Tensor) else tensor
    return Tensor(np.full_like(a=array, fill_value=value))


def expand_tensor(tensor: Tensor, shape: Shape) -> Tensor:
    """Expand the shape of a tensor to a broadcastable shape.

    Parameters
    ----------
    tensor : Tensor
    shape : Shape

    Returns
    -------
    Tensor

    """

    array = tensor.array

    # If the shapes already align, do nothing
    if array.shape != shape:
        # If the size of the array is the same as the shape, then just reshape
        if array.size == np.prod(shape):
            array = np.reshape(array, shape)
        # Otherwise, try to broadcast the array to the shape
        # Do nothing if it fails
        else:
            try:
                array = np.array(np.broadcast_to(array, shape))
            except:
                pass

    return Tensor(array)


def shrink_tensor(tensor: Tensor, shape: Shape) -> Tensor:
    """Shrink the shape of a tensor from a broadcastable shape.

    Parameters
    ----------
    tensor : Tensor
    shape : Shape

    Returns
    -------
    Tensor

    """

    array = tensor.array

    # If the shapes already align, do nothing
    if array.shape != shape:
        # If the size of the array is the same as the shape, then just reshape
        if array.size == np.prod(shape):
            array = np.reshape(array, shape)
        else:
            # If the broadcastable shape is a scalar, then take the full mean
            if len(shape) < 1:
                array = np.sum(array).reshape(shape)
            # Otherwise, try to broadcast the array to the shape
            # Do nothing if it fails
            else:
                try:
                    broad_shape = np.broadcast_shapes(array.shape, shape)
                except:
                    pass
                else:
                    # Get intermediate broadcast shape according to numpy broadcasting rules
                    inter_shape = [1] * (len(broad_shape) - len(shape)) + list(shape)
                    # Get the axis indices that are not the same
                    axes = []
                    for i in reversed(range(len(broad_shape))):
                        if array.shape[i] != inter_shape[i]:
                            axes.append(i)
                    # Take the mean across the axes that are not the same to collect values
                    array = np.sum(array, axis=tuple(axes)).reshape(shape)

    return Tensor(array)
