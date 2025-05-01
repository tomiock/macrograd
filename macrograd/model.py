from typing import Optional
import numpy as np
from macrograd.engine import Graph, get_default_graph
from macrograd.tensor import Tensor


class Linear:
    def __init__(
        self,
        in_dims,
        out_dims,
        graph: Optional[Graph] = None,
        initilization: Optional[str] = None,
    ):
        self.in_dims = in_dims
        self.out_dims = out_dims
        self._w_shape = (in_dims, out_dims)
        self._b_shape = (1, out_dims)
        self.initilization = initilization

        if graph is None:
            graph = get_default_graph()
        self.graph = graph

        self.w = None
        self.b = None

    def __call__(self, data):
        data = data @ self.w
        return data + self.b

    def init_params(self):
        self.w = Tensor(
            np.random.randn(*self._w_shape),
            requires_grad=True,
            graph=self.graph,
        )

        # TODO: small positive, so everything moves to the linear side of the relu
        self.b = Tensor(
            np.zeros(self._b_shape),
            requires_grad=True,
            graph=self.graph,
        )

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


class SGD_Optimizer:
    def __init__(
        self, learning_rate: float = 0.01, minimizing=True, step_function=None
    ):
        self.lr = learning_rate
        self.min = minimizing
        self.step_fn = step_function

        if not self.step_fn:
            if self.min:

                def step_fn(param):
                    param.data -= param.grad * self.lr
                    return param
            else:

                def step_fn(param):
                    param.data += param.grad * self.lr
                    return param

            self.step_fn = step_fn

    def step(self, loss: Tensor, params: list[Tensor]) -> list[Tensor]:
        loss.backprop()

        return list(map(self.step_fn, params))


class SGD_MomentumOptimizer:
    def __init__(
        self, learning_rate: float = 0.01, alpha: float = 0.9, params_copy: list = []
    ):
        self.lr = learning_rate
        self.velocities = []

        if 0 <= alpha < 1:
            self.alpha = Tensor(alpha, requires_grad=False)
        else:
            raise ValueError("Alpha hyperparemter must be between 0 and 1")

        for param in params_copy:
            self.velocities.append(
                Tensor(
                    np.zeros_like(param.data),
                    requires_grad=False,
                )
            )

        del params_copy

    def step_fn(self, param: Tensor, index: int):
        self.velocities[index].data = (
            self.alpha.data * self.velocities[index].data - self.lr * param.grad
        )
        param.data = param.data + self.velocities[index].data
        return param

    def step(self, loss: Tensor, params: list[Tensor]) -> list[Tensor]:
        loss.backprop()

        updated_params = []
        for index, param in enumerate(params):
            updated_params.append(self.step_fn(param, index))
        return updated_params
