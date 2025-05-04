from math import prod

import numpy as np
from .tensor import Tensor


def BCE(y_pred: Tensor, y_true) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = y_true.reshape((-1, 1))
        y_true = Tensor(array=y_true)

    y_true.requires_grad = False

    loss_val = y_true * y_pred.log(base=2) + (1 - y_true) * 1 - y_pred.log(base=2)
    size_y_true = prod(y_true.node.shape)

    return -1 * (loss_val.sum() / size_y_true)


def sigmoid(x: Tensor):
    return x / (1 + (-x).exp())

def relu(x: Tensor):
    x_shape = x.node.shape
    return maximum(x, np.zeros_like(x_shape))

def maximum(tensor1, tensor2):
    pass

