import numpy as np
import warnings

from . import Tensor


class LinearScheduler:
    def __init__(self, total_iter, target_lr):
        self.total_iter = total_iter
        self.target_lr = target_lr

        self.step_count = 0

    def step(self, optimizer):
        try:
            assert isinstance(optimizer.lr, float)
        except NameError or TypeError or AssertionError:
            raise Exception("Provide a valid optimizer or learning rate")
        lr = optimizer.lr

        new_lr = self.get_lr(self.step_count, self.total_iter, lr, self.target_lr)
        optimizer.lr = new_lr

        self.step_count += 1
        return optimizer

    def get_lr(self, current_epoch, final_epoch, learning_rate, final_learning_rate):
        if current_epoch < final_epoch:
            alpha = current_epoch / final_epoch
            return (1 - alpha) * learning_rate + alpha * final_learning_rate
        else:
            return final_learning_rate


class ExponentailScheduler:
    def __init__(self, decay_factor, decay_rate):
        self.decay_factor = decay_factor
        self.decay_rate = decay_rate
        self.step_count = 0

    def step(self, optimizer):
        try:
            assert isinstance(optimizer.lr, float)
        except NameError or TypeError or AssertionError:
            raise Exception("Provide a valid optimizer or learning rate")

        raise NotImplementedError


class StepScheduler:
    def __init__(self, decay_factor: float, decay_rate=2):
        self.decay_factor = decay_factor
        self.decay_rate = decay_rate
        self.step_count = 0
        warnings.warn(
            "The learning rate eventually would be zero, set the decay rate carefully"
        )

    def step(self, optimizer):
        try:
            assert isinstance(optimizer.lr, float)
        except NameError or TypeError or AssertionError:
            raise Exception("Provide a valid optimizer or learning rate")

        lr = optimizer.lr
        optimizer.lr = self.get_lr(self.step_count, lr)

        self.step_count += 1
        return optimizer

    # WARNING: the lr eventually would be zero
    def get_lr(self, current_epoch, learning_rate):
        if not current_epoch % self.decay_rate:
            return learning_rate * self.decay_factor
        else:
            return learning_rate


class ReduceLROnPlateau:
    def __init__(self, optimizer):
        pass

    def step(self, learning_rate: float, metric_history: list):
        pass


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
        self,
        params_copy: list,
        learning_rate: float = 0.01,
        alpha: float = 0.9,
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

        loss.backprop()

        updated_params = []
        for index, param in enumerate(params):
            updated_params.append(self.step_fn(param, index))
        return updated_params
