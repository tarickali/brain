"""
title : optimizer.py
create : @tarickali 23/12/18
update : @tarickali 23/12/18
"""

from abc import ABC, abstractmethod
from brain.core import Node


class Optimizer(ABC):
    def __init__(self, parameters: list[dict[str, Node]], learning_rate: float) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.time = 0

    def increment(self) -> None:
        self.time += 1

    def reset(self) -> None:
        self.time = 0

    @abstractmethod
    def update(self) -> None:
        """ """

        raise NotImplementedError
