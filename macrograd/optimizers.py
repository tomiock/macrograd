import numpy as np

from . import Tensor


class SGD:
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
        for param in params:
            param.zero_grad()

        loss.backprop()

        return list(map(self.step_fn, params))


class SGD_Momentum:
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
                    np.zeros_like(param.data), requires_grad=False, precision=np.float32
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
        for param in params:
            param.zero_grad()

        loss.backprop()

        updated_params = []
        for index, param in enumerate(params):
            updated_params.append(self.step_fn(param, index))
        return updated_params
