"""
title : activation.py
create : @tarickali 23/12/15
update : @tarickali 23/12/17
"""

from abc import ABC, abstractmethod
from brain.core import Node


class Activation(ABC):
    @abstractmethod
    def func(self, x: Node) -> Node:
        """Compute the output of the activation function of the input x.

        Parameters
        ----------
        x : Node

        Returns
        -------
        Node

        """

        raise NotImplementedError

    def __call__(self, x: Node) -> Node:
        """Compute the output of the activation function of the input x.

        Parameters
        ----------
        x : Node

        Returns
        -------
        Node

        """

        return self.func(x)
