"""
title : types.py
create : @tarickali 23/12/13
update : @tarickali 23/12/20
"""

from numpy import ndarray, dtype, number

__all__ = ["Array", "Number", "ArrayLike", "Dtype", "Shape"]

Array = ndarray
Number = number | int | float | bool
ArrayLike = Array | list | Number
Dtype = dtype | int | float | bool
Shape = tuple[None | int, ...]
