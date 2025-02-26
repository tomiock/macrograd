from collections.abc import Callable
import numpy as np
from macrograd.tensor import Tensor


class Linear:
    def __init__(self, in_dims, out_dims):
        self.in_dims = in_dims
        self.out_dims = out_dims
        self._w_shape = (in_dims, out_dims)
        self._b_shape = (1, out_dims)

        self.w = None
        self.b = None

    def __call__(self, data):
        return (data @ self.w) + self.b

    def init_params(self):
        self.w = Tensor(np.random.randn(*self._w_shape), requires_grad=True)
        self.b = Tensor(np.random.randn(*self._b_shape), requires_grad=True)

        return [self.w, self.b]

    def set_params(self, w, b):  # Added set_params
        self.w = w
        self.b = b


class Model:
    def __init__(self):
        self.layers = []
        self._params = []

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Linear):
            self.layers.append(value)

    @property
    def parameters(self):
        return self._params

    @parameters.setter
    def parameters(self, params: list[Tensor]):
        self._params = params

    def init_params(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.init_params())
        self._params = params
        return self._params


class Optimizer:
    def __init__(
        self, learning_rate: float = 0.01, minimizing=True, step_function=None
    ):
        self.lr = learning_rate
        self.min = minimizing
        self.step_fn = step_function

        if not self.step_fn:
            if self.min:

                def step_fn(param):
                    param.arr -= param.grad * self.lr
                    return param
            else:

                def step_fn(param):
                    param.arr += param.grad * self.lr
                    return param

            self.step_fn = step_fn

    def step(self, loss: Tensor, params: list[Tensor]) -> list[Tensor]:
        for param in params:
            param.zero_grad()

        loss.backprop()

        return list(map(self.step_fn, params))
