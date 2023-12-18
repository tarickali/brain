"""
title : loss.py
create : @tarickali 23/12/18
update : @tarickali 23/12/18
"""

from abc import ABC, abstractmethod
from brain.core import Node


class Loss(ABC):
    @abstractmethod
    def loss(self, y_true: Node, y_pred: Node) -> Node:
        """Computes the loss value between the ground truth and predicted arrays.

        Parameters
        ----------
        y_true : Node
        y_pred : Node

        Returns
        -------
        float

        """

        raise NotImplementedError

    def __call__(self, y_true: Node, y_pred: Node) -> Node:
        """Computes the loss value between the ground truth and predicted arrays.

        Parameters
        ----------
        y_true : Node
        y_pred : Node

        Returns
        -------
        float

        """

        return self.loss(y_true, y_pred)
