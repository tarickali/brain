"""
title : factories.py
create : @tarickali 23/12/15
update : @tarickali 23/12/17
"""

from typing import Any
from brain.core import Initializer, Activation
from brain.initializers import *
from brain.activations import *


def initializer_factory(
    initializer: str | dict[str, Any] | Initializer = None
) -> Initializer:
    """Factory function to create Initializer objects.

    If initializer = None, returns the RandomNormal initializer.

    Parameters
    ----------
    initializer : str | dict[str, Any] | Initializer = None
        If str, then represents the name of the Initializer.
        If dict, then represents the name, param dict of the Initializer.
        If Initializer, then returns the same object.

    Returns
    -------
    Initializer

    Raises
    ------
    ValueError
        If the given initializer is not recognized.

    """

    if initializer is None:
        name = "random_normal"
        params = {}
    elif isinstance(initializer, str):
        name = initializer
        params = {}
    elif isinstance(initializer, dict):
        name = initializer["name"]
        params = initializer["params"]
    elif isinstance(initializer, Initializer):
        return initializer
    else:
        raise ValueError("Cannot interpret given activation for factory.")

    match name:
        case "constant":
            return Constant(**params)
        case "he_normal":
            return HeNormal()
        case "he_uniform":
            return HeUniform()
        case "lecun_normal":
            return LecunNormal()
        case "lecun_uniform":
            return LecunUniform()
        case "ones":
            return Ones()
        case "random_normal":
            return RandomNormal(**params)
        case "random_uniform":
            return RandomUniform(**params)
        case "xavier_normal":
            return XavierNormal()
        case "xavier_uniform":
            return XavierUniform()
        case "zeros":
            return Zeros()
        case _:
            raise ValueError(f"Initializer: {name} not available.")


def activation_factory(
    activation: str | dict[str, Any] | Activation = None
) -> Activation:
    """Factory function to create Activation objects.

    If activation = None, returns the Identity activation.

    Parameters
    ----------
    activation : str | dict[str, Any] | Activation = None
        If str, then represents the name of the Activation.
        If dict, then represents the name, param dict of the Activation.
        If Activation, then returns the same object.

    Returns
    -------
    Activation

    Raises
    ------
    ValueError
        If the given activation is not recognized.

    """

    if activation is None:
        name = "identity"
        params = {}
    elif isinstance(activation, str):
        name = activation
        params = {}
    elif isinstance(activation, dict):
        name = activation["name"]
        params = activation["params"]
    elif isinstance(activation, Activation):
        return activation
    else:
        raise ValueError("Cannot interpret given activation for factory.")

    match name:
        case "affine":
            return Affine(**params)
        case "elu":
            return ELU(**params)
        case "identity":
            return Identity()
        case "relu":
            return ReLU()
        case "selu":
            return SELU()
        case "sigmoid":
            return Sigmoid()
        case "softplus":
            return SoftPlus()
        case "tanh":
            return Tanh()
        case _:
            raise ValueError(f"Activation: {name} not available.")
