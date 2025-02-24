from tomi_grad.tensor import Tensor


class Model:
    def __init__(self):
        pass

    @property
    def parameters(self):
        pass

    @parameters.setter
    def parameters(self, params: tuple[Tensor, ...]):
        self.parameters = params

    def init_params(self) -> tuple[Tensor, ...]:
        raise NotImplementedError

    def forward(self, data: Tensor) -> Tensor:
        raise NotImplementedError


class Optimizer:
    def __init__(self):
        pass

    def step(self, loss: Tensor, params: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return params


def get_func_model(model: Model):
    def fn(data: Tensor, params: tuple[Tensor, ...]) -> Tensor:
        model.parameters = params
        return model.forward(data)

    return fn


"""
model = Model()
opt = Optimizer()
loss = Loss()

model_func = get_func_model(model)

params = model.init_params()

for _ in epochs:
    y = model_func(X, params)
    loss_value = loss(y, y_train)
    params = optimizer.step(loss_value, params)

"""
