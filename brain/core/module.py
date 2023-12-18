"""
title : module.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

from typing import Any
from abc import ABC, abstractmethod

from brain.core import Node

__all__ = ["Module"]


class Module(ABC):
    """ """

    def __init__(self) -> None:
        super().__init__()
        self.parameters: dict[str, Node] = {}
        self.trainable: bool = True

    @abstractmethod
    def forward(self, X: Node) -> Node:
        """Computes the forward pass of the Module on input X.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray

        """

        raise NotImplementedError

    def init_parameters(self) -> None:
        """Initialize the parameters of the Module."""

        return

    def zero_gradients(self) -> None:
        """Clear the gradients for each parameter in the Module."""

        for param in self.parameters:
            self.parameters[param].zero_grad()

    def freeze(self) -> None:
        """Set the Module to be untrainable."""

        for param in self.parameters:
            self.parameters[param].trainable = False
        self.trainable = False

    def unfreeze(self) -> None:
        """Set the Module to be trainable."""

        for param in self.parameters:
            self.parameters[param].trainable = True
        self.trainable = True

    def summary(self) -> dict[str, Any]:
        """Get a summary of the Module.

        Returns
        -------
        dict[str, Any]

        """

        return {
            "name": self.name,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]:
        """Get the hyperparameters of the Module.

        Returns
        -------
        dict[str, Any]

        """

        raise NotImplementedError
