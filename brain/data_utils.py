"""
title : data_utils.py
create : @tarickali 23/12/19
update : @tarickali 23/12/19
"""

import numpy as np

__all__ = ["one_hot", "get_batches"]


def one_hot(x: np.ndarray, k: int = 10) -> np.ndarray:
    """Create a one-hot array of input array with k classes.

    Parameters
    ----------
    x : np.ndarray @ (n, 1)
    k : int = 10
        Number of classes in the one-hot array.

    Returns
    -------
    np.ndarray @ (n, k)

    """

    n = x.shape[0]
    o = np.zeros((n, k))
    o[np.arange(n), x] = 1
    return o


def get_batches(
    X: np.ndarray, y: np.ndarray, m: int = 32
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create batches of (X, y) data.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    m : int = 32

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]

    """

    n = X.shape[0]
    batches = []

    # Loop for creating batches of size m
    for i in range(n // m):
        a, b = i * m, (i + 1) * m
        batches.append((X[a:b], y[a:b]))

    # Create an extra match of size < m for leftover data
    if b != n:
        batches.append((X[b:], y[b:]))

    return batches
