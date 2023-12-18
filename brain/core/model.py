"""
title : model.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

from abc import ABC, abstractmethod

from brain.core import Node


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: Node) -> Node:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> list[dict[str, Node]]:
        raise NotImplementedError

    def __call__(self, X: Node) -> Node:
        return self.forward(X)
