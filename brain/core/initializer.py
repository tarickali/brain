"""
title : initializer.py
create : @tarickali 23/12/14
update : @tarickali 23/12/14
"""

from abc import ABC, abstractmethod

from brain.core.types import Shape
from brain.core.tensor import Tensor


class Initializer(ABC):
    @abstractmethod
    def init(self, shape: Shape) -> Tensor:
        """Initialize a Tensor with given shape using initialization scheme.

        Parameters
        ----------
        shape : Shape

        Returns
        -------
        Tensor

        """

        raise NotImplementedError

    def __call__(self, shape: Shape) -> Tensor:
        """Initialize a Tensor with given shape using initialization scheme.

        Parameters
        ----------
        shape : Shape

        Returns
        -------
        Tensor

        """

        return self.init(shape)
