"""
title : types.py
create : @tarickali 23/12/13
update : @tarickali 23/12/17
"""

from numpy import ndarray, dtype, number

__all__ = ["Array", "Numeric", "Dtype", "Shape"]

Array = ndarray | list
Numeric = number | int | float | bool
Dtype = dtype | int | float | bool
Shape = tuple[None | int, ...]
