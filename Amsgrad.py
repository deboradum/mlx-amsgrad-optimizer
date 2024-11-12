# Unofficial MLX implementation of Amsgrad optimizer
# https://arxiv.org/abs/1904.09237
#
#  Written by Pepijn van Wijk
# https://github.com/deboradum
#
# Apache-2.0 license

from typing import Callable, List, Union

import mlx.core as mx
from mlx.optimizers import Optimizer


class Amsgrad(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 1e-3,
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-6,
        beta_decay: bool = False,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps
        self.beta_decay = beta_decay

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["v_hat"] = mx.zeros_like(parameter)
        state["t"] = 1

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the ADOPT parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        eps = self.eps

        m = state["m"]
        v = state["v"]
        v_hat = state["v_hat"]
        t = state["t"]

        if self.beta_decay:
            b1 = b1 / t

        m = b1 * m + (1 - b1) * gradient

        v = b2 * v + (1 - b2) * mx.square(gradient)
        v_hat = mx.maximum(v_hat, v)

        state["m"] = m
        state["v"] = v
        state["v_hat"] = v_hat
        state["t"] = t + 1

        return parameter - lr * (m / (mx.sqrt(v_hat) + eps))
