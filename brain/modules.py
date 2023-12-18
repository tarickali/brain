"""
title : modules.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

from typing import Any
from brain.core import Node, Module
from brain.factories import activation_factory, initializer_factory
from brain.utils import flatten

__all__ = ["Linear", "Flatten", "Activation"]


class Linear(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str | dict[str, Any] = None,
        weight_initializer: str | dict[str, Any] = "xavier_normal",
        bias_initializer: str | dict[str, Any] = "zeros",
        include_bias: bool = True,
        name: str = "Linear",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.include_bias = include_bias
        self.name = name

        self.act_fn = activation_factory(activation)
        self.weight_init = initializer_factory(weight_initializer)
        if self.include_bias:
            self.bias_init = initializer_factory(bias_initializer)

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize the parameters of the Module."""

        self.parameters["W"] = Node(self.weight_init((self.input_dim, self.output_dim)))
        assert self.parameters["W"].shape == (self.input_dim, self.output_dim)

        if self.include_bias:
            self.parameters["b"] = Node(self.bias_init((self.output_dim,)))
            assert self.parameters["b"].shape == (self.output_dim,)

    def forward(self, X: Node) -> Node:
        # Get weights and bias
        W, b = self.parameters["W"], self.parameters.get("b", None)

        # Compute linear transformation
        if b is None:
            Z = X @ W
        else:
            Z = X @ W + b

        assert Z.shape == (X.shape[0], self.output_dim)

        # Compute activation
        A = self.act_fn(Z)

        return A

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "weight_initializer": self.weight_initializer,
            "bias_initializer": self.bias_initializer,
            "include_bias": self.include_bias,
        }


class Flatten(Module):
    def __init__(self, name: str = "Flatten") -> None:
        super().__init__()
        self.name = name

    def forward(self, X: Node) -> Node:
        return flatten(X)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {}


class Activation(Module):
    def __init__(
        self, activation: str | dict[str, Any], name: str = "Activation"
    ) -> None:
        super().__init__()
        self.activation = activation
        self.name = name

        self.act_fn = activation_factory(activation)

    def forward(self, X: Node) -> Node:
        return self.act_fn(X)

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {"activation": self.activation}
