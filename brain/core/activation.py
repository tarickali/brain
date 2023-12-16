"""
title : activation.py
create : @tarickali 23/12/15
update : @tarickali 23/12/15
"""

from abc import ABC, abstractmethod
from brain.core import Tensor


class Activation(ABC):
    @abstractmethod
    def func(self, x: Tensor) -> Tensor:
        """Compute the output of the activation function of the input x.

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor

        """

        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        """Compute the output of the activation function of the input x.

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor

        """

        return self.func(x)
