"""
title : models.py
create : @tarickali 23/12/17
update : @tarickali 23/12/19
"""

from brain.core import Node, Module, Model

__all__ = ["Sequential"]


class Sequential(Model):
    def __init__(self, modules: list[Module] = None) -> None:
        self.modules = modules

    def forward(self, X: Node) -> Node:
        for module in self.modules:
            X = module(X)
        return X

    def zero_grad(self) -> None:
        for module in self.modules:
            module.zero_grad()

    def append(self, module: Module) -> None:
        """Add a Module to the end of the Sequential Model.

        Parameters
        ----------
        module : Module

        """

        self.modules.append(module)

    @property
    def parameters(self) -> list[dict[str, Node]]:
        return [module.parameters for module in self.modules]
