"""
title : types.py
create : @tarickali 23/12/13
update : @tarickali 23/12/13
"""

from numpy import ndarray, dtype, number

Array = ndarray | list
Numeric = number | int | float | bool
Dtype = dtype | int | float | bool
Shape = tuple[None | int, ...]
