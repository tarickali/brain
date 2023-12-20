"""
title : types.py
create : @tarickali 23/12/13
update : @tarickali 23/12/20
"""

from numpy import ndarray, dtype, number

__all__ = ["Array", "List", "Number", "ArrayLike", "Dtype", "Shape"]

Array = ndarray
List = list
Number = number | int | float | bool
ArrayLike = Array | List | Number
Dtype = dtype | int | float | bool
Shape = tuple[None | int, ...]
