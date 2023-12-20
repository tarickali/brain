"""
title : tensor.py
create : @tarickali 23/12/13
update : @tarickali 23/12/20
"""

from __future__ import annotations
from typing import Any
import numpy as np
from brain.core.types import Array, Number, ArrayLike, Dtype, Shape

__all__ = ["Tensor"]

TensorLike = ArrayLike


class Tensor:
    def __init__(self, array: Tensor | TensorLike, dtype: Dtype = float) -> None:
        array = array.array if isinstance(array, Tensor) else array
        self.array: np.ndarray = np.array(array, dtype=dtype)

    def cast(self, dtype: Dtype) -> None:
        if dtype != self.dtype:
            self.array = self.array.astype(dtype)

    def transpose(self) -> Tensor:
        return Tensor(self.array.T)

    def numpy(self) -> Array:
        return self.array

    def item(self) -> Number:
        return self.array.item()

    # ---------------------------------------------------------#
    #################### Binary Operations ####################
    # ---------------------------------------------------------#

    def __add__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array + other.array)

    def __sub__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array - other.array)

    def __mul__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array * other.array)

    def __matmul__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array @ other.array)

    def __truediv__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array / other.array)

    def __radd__(self, other: Tensor | TensorLike) -> Tensor:
        return self + other

    def __rsub__(self, other: Tensor | TensorLike) -> Tensor:
        return -self + other

    def __rmul__(self, other: Tensor | TensorLike) -> Tensor:
        return self * other

    def __rtruediv__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=other.array / self.array)

    # ---------------------------------------------------------#
    ##################### Unary Operations #####################
    # ---------------------------------------------------------#

    def __pow__(self, other: Number) -> Tensor:
        if not isinstance(other, Number):
            raise ValueError(f"Cannot perform operation on {type(other)}")
        return Tensor(array=self.array**other)

    def __neg__(self) -> Tensor:
        return Tensor(array=-self.array)

    # ---------------------------------------------------------#
    ################## Comparison Operations ##################
    # ---------------------------------------------------------#

    def __eq__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array == other.array)

    def __ne__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array != other.array)

    def __ge__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array >= other.array)

    def __gt__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array > other.array)

    def __le__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array <= other.array)

    def __lt__(self, other: Tensor | TensorLike) -> Tensor:
        other = convert_tensor_input(other)
        return Tensor(array=self.array < other.array)

    def __repr__(self) -> str:
        return f"Tensor({self.array}, dtype={self.dtype}, shape={self.shape})"

    @property
    def T(self) -> Tensor:
        return Tensor(self.array.T)

    @property
    def shape(self) -> Shape:
        return self.array.shape

    @property
    def dtype(self) -> Shape:
        return self.array.dtype

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def size(self) -> int:
        return self.array.size


def convert_tensor_input(value: Any) -> Tensor:
    if not isinstance(value, Tensor | TensorLike):
        raise ValueError(f"Cannot perform operation on {type(value)}")
    return value if isinstance(value, Tensor) else Tensor(value)
