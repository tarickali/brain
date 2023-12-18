"""
title : constants.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

import numpy as np
import sys

__all__ = ["eps", "e", "pi", "maxsize", "minsize"]

eps = np.finfo(float).eps
e = np.e
pi = np.pi
maxsize = sys.maxsize
minsize = -sys.maxsize - 1
