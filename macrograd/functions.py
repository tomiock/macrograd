import numpy as np

from .tensor import Tensor, _to_var
from .tensor import get_axes_broadcasting


def sin(A: Tensor):
    result = Tensor(np.sin(A.data))
    result.requires_grad = A.requires_grad

    if A.requires_grad:

        def _grad_a(_value):
            return np.cos(_value)

        result.parents.add((A, _grad_a))

    result._nodes_edges.add(A)
    return result


def cos(A: Tensor):
    result = Tensor(np.cos(A.data))
    result.requires_grad = A.requires_grad

    if A.requires_grad:

        def _grad_a(_value):
            return -np.sin(_value)

        result.parents.add((A, _grad_a))

    result._nodes_edges.add(A)
    return result


def ln(A: Tensor):
    result = Tensor(np.log(A.data), _op="ln")
    result.requires_grad = A.requires_grad

    if A.requires_grad:

        def _grad_a(_value):
            return _value * (1.0 / A.data)

        result.parents.add((A, _grad_a))

    result._nodes_edges.add(A)
    return result


def log2(A: Tensor):
    result = Tensor(np.log2(A.data), _op="log2")
    result.requires_grad = A.requires_grad

    if A.requires_grad:

        def _grad_a(_value):
            return _value * (1.0 / (A.data * np.log(2.0)))

        result.parents.add((A, _grad_a))

    result._nodes_edges.add(A)
    # print(f"{result.shape} = {result._op}({A.label})")
    return result


def MSE(x_1: Tensor, x_2: Tensor):
    assert x_1.shape == x_2.shape

    n = x_1.data.size  # Get the number of elements (for averaging)
    diff = x_1 - x_2
    result = (diff**2).sum_np() / n  # Divide by n here

    if x_1.requires_grad:

        def _grad_x1(incoming_grad):
            local_grad = (2.0 / n) * (x_1.data - x_2.data)
            sum_axes = get_axes_broadcasting(incoming_grad, x_1.shape)
            return np.sum(
                incoming_grad * local_grad, axis=tuple(sum_axes), keepdims=True
            )

        result.parents.add((x_1, _grad_x1))

    if x_2.requires_grad:

        def _grad_x2(incoming_grad):
            local_grad = (2.0 / n) * (x_2.data - x_1.data)
            sum_axes = get_axes_broadcasting(incoming_grad, x_2.shape)
            return np.sum(
                incoming_grad * local_grad, axis=tuple(sum_axes), keepdims=True
            )

        result.parents.add((x_2, _grad_x2))

    return result


def BCE(x: Tensor, y: Tensor) -> Tensor:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    n = x.data.size  # Total number of elements (for averaging)
    epsilon = 1e-15  # Small value for clipping

    x_clipped = np.clip(x.data, epsilon, 1.0 - epsilon)

    loss_val = -(y.data * np.log(x_clipped) + (1 - y.data) * np.log(1 - x_clipped))
    result = Tensor(
        np.sum(loss_val) / n, requires_grad=(x.requires_grad or y.requires_grad)
    )

    if x.requires_grad:

        def _grad_x(incoming_grad):
            local_grad = (x_clipped - y.data) / (x_clipped * (1.0 - x_clipped))
            sum_axes = get_axes_broadcasting(incoming_grad, x.shape)
            return np.sum(
                incoming_grad * local_grad, axis=tuple(sum_axes), keepdims=True
            )

        result.parents.add((x, _grad_x))
        # x.children.append(result)

    return result


def sigmoid(A: Tensor):
    A = _to_var(A)
    result = Tensor(1.0 / (1.0 + np.exp(-A.data)), _op="sigmoid")
    result.requires_grad = A.requires_grad

    if result.requires_grad:

        def _grad_a(_value):
            sig = 1.0 / (1.0 + np.exp(-A.data))
            local_grad = sig * (1 - sig)
            return _value * local_grad

        result.parents.add((A, _grad_a))
    result._nodes_edges.add(A)
    return result


def relu(A: Tensor):
    A = _to_var(A)
    result = Tensor(np.maximum(0, A.data), _op="relu")
    result.requires_grad = A.requires_grad

    if result.requires_grad:

        def _grad_relu(_value):
            local_grad = np.array((A.data > 0)).astype(A.data.dtype)
            return _value * local_grad

        result.parents.add((A, _grad_relu))

    result._nodes_edges.add(A)
    return result


def cross_entropy(y_true, y_pred):
    return -1 * (y_true * log2(y_pred)).sum() / y_true.data.shape[0]


def softmax(x: Tensor):
    e_x = e**x
    return e_x / (e_x.sum(axis=1, keepdims=True))
