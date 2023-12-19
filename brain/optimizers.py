"""
title : optimizers.py
create : @tarickali 23/12/18
update : @tarickali 23/12/18
"""

import numpy as np
from brain.core import Tensor, Node, Optimizer

__all__ = ["SGD", "Adadelta", "Adagrad", "Adam", "RMSprop"]


class SGD(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Node]],
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize

        self.cache: list[dict[str, dict[str, Tensor]]] = [
            {"velocity": {}} for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        L = len(self.parameters)
        t = self.time + 1
        for l in range(L):
            params = self.parameters[l]
            cache = self.cache[l]

            for key, param in params.items():
                g = param.grad + self.weight_decay * param.data

                if self.momentum != 0.0:
                    # Initialize the cache values
                    if t == 1:
                        cache["velocity"][key] = g
                    else:
                        cache["velocity"][key] = (
                            self.momentum * cache["velocity"][key] + self.dampening * g
                        )

                    if self.nesterov:
                        g = g + self.momentum * cache["velocity"][key]
                    else:
                        g = cache["velocity"][key]

                if self.maximize:
                    param.data = param.data + self.learning_rate * g
                else:
                    param.data = param.data - self.learning_rate * g

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [{"velocity": {}} for _ in range(len(self.parameters))]


class Adadelta(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Node]],
        learning_rate: float = 1.0,
        rho: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.rho = rho
        self.weight_decay = weight_decay
        self.eps = eps
        self.maximize = maximize

        self.cache: list[dict[str, dict[str, Tensor]]] = [
            {"average": {}, "accumulator": {}} for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        L = len(self.parameters)
        t = self.time + 1
        for l in range(L):
            params = self.parameters[l]
            cache = self.cache[l]

            for key, param in params.items():
                g = param.grad + self.weight_decay * param.data

                # Initialize the cache values
                if t == 1:
                    cache["average"][key] = (1 - self.rho) * g**2
                    delta = ((self.eps) / (cache["average"][key] + self.eps)) ** 0.5 * g
                    cache["accumulator"][key] = (1 - self.rho) * delta**2
                else:
                    cache["average"][key] = (
                        self.rho * cache["average"][key] + (1 - self.rho) * g**2
                    )
                    delta = (
                        (cache["accumulator"][key] + self.eps)
                        / (cache["average"][key] + self.eps)
                    ) ** 0.5 * g
                    cache["accumulator"][key] = (
                        self.rho * cache["accumulator"][key]
                        + (1 - self.rho) * delta**2
                    )

                if self.maximize:
                    param.data = param.data + self.learning_rate * delta
                else:
                    param.data = param.data - self.learning_rate * delta

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [{"sum": {}} for _ in range(len(self.parameters))]


class Adagrad(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Node]],
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.maximize = maximize

        self.cache: list[dict[str, dict[str, Tensor]]] = [
            {"sum": {}} for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        L = len(self.parameters)
        t = self.time + 1
        for l in range(L):
            params = self.parameters[l]
            cache = self.cache[l]

            for key, param in params.items():
                g = param.grad + self.weight_decay * param.data
                lr = self.learning_rate / (1 + (t - 1) * self.learning_rate_decay)

                # Initialize the cache values
                if t == 1:
                    cache["sum"][key] = Tensor(
                        np.full_like(param.data, self.initial_accumulator_value)
                    )
                cache["sum"][key] = cache["sum"][key] + g**2

                assert g.shape == cache["sum"][key].shape == param.data.shape

                param.data = param.data - lr * g / (cache["sum"][key] ** 0.5 + self.eps)
                # assert isinstance(param.data, Tensor)

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [{"sum": {}} for _ in range(len(self.parameters))]


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Node]],
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        self.cache: list[dict[str, dict[str, Tensor]]] = [
            {"momentum": {}, "velocity": {}, "vhat_max": {}}
            for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        L = len(self.parameters)
        t = self.time + 1
        for l in range(L):
            params = self.parameters[l]
            cache = self.cache[l]

            for key, param in params.items():
                g = param.grad if not self.maximize else -param.grad
                g = g + self.weight_decay * param.data

                # Initialize the cache values
                if t == 1:
                    cache["momentum"][key] = (1 - self.beta_1) * g
                    cache["velocity"][key] = (1 - self.beta_2) * g**2
                else:
                    cache["momentum"][key] = (
                        self.beta_1 * cache["momentum"][key] + (1 - self.beta_1) * g
                    )
                    cache["velocity"][key] = (
                        self.beta_2 * cache["velocity"][key]
                        + (1 - self.beta_2) * g**2
                    )

                mhat = cache["momentum"][key] / (1 - self.beta_1**t)
                vhat = cache["velocity"][key] / (1 - self.beta_2**t)

                if self.amsgrad:
                    if t == 1:
                        cache["vhat_max"][key] = Tensor(vhat.array)
                    else:
                        cache["vhat_max"][key] = Tensor(
                            np.maximum(cache["vhat_max"][key].array, vhat.array)
                        )  # NOTE: bug here?
                    param.data = param.data - self.learning_rate * mhat / (
                        np.sqrt(cache["vhat_max"][key].array) + self.eps
                    )
                else:
                    param.data = param.data - self.learning_rate * mhat / (
                        vhat.array**0.5 + self.eps
                    )

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.momentums = [{} for _ in range(len(self.parameters))]
        self.velocities = [{} for _ in range(len(self.parameters))]


class RMSprop(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Node]],
        learning_rate: float = 0.01,
        alpha: float = 0.99,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        centered: bool = False,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.alpha = alpha
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.centered = centered
        self.eps = eps
        self.maximize = maximize

        self.cache: list[dict[str, dict[str, Tensor]]] = [
            {"square_average": {}, "buffer": {}, "g_av": {}}
            for _ in range(len(self.parameters))
        ]

    def update(self) -> None:
        L = len(self.parameters)
        t = self.time + 1
        for l in range(L):
            params = self.parameters[l]
            cache = self.cache[l]

            for key, param in params.items():
                g = param.grad + self.weight_decay * param.data

                # Initialize the cache values
                if t == 1:
                    cache["square_average"][key] = Tensor(np.zeros_like(g))
                    cache["buffer"][key] = Tensor(np.zeros_like(g))
                    cache["g_av"][key] = Tensor(np.zeros_like(g))

                cache["square_average"][key] = (
                    self.alpha * cache["square_average"][key]
                    + (1 - self.alpha) * g**2
                )
                v_hat = cache["square_average"][key]

                if self.centered:
                    cache["g_av"][key] = (
                        cache["g_av"][key] * self.alpha + (1 - self.alpha) * g
                    )
                    v_hat = v_hat - cache["g_av"][key] ** 2

                if self.momentum > 0:
                    cache["buffer"][key] = self.momentum * cache["buffer"][key] + g / (
                        np.sqrt(v_hat.array) + self.eps
                    )
                    param.data = param.data - self.learning_rate * cache["buffer"][key]
                else:
                    param.data = param.data - self.learning_rate * g / (
                        np.sqrt(v_hat.array) + self.eps
                    )

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"square_average": {}, "buffer": {}, "g_av": {}}
            for _ in range(len(self.parameters))
        ]
