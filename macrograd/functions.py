import numpy as np

from .tensor import Tensor, _to_var
from .tensor import get_axes_broadcasting


def sin(A: Tensor):
    result = Tensor(np.sin(A.arr))
    result.requires_grad = A.requires_grad

    if A.requires_grad:

        def _grad_a(_value):
            return np.cos(_value)

        result.parents.append((A, _grad_a))

    return result


def cos(A: Tensor):
    result = Tensor(np.cos(A.arr))
    result.requires_grad = A.requires_grad

    if A.requires_grad:

        def _grad_a(_value):
            return -np.sin(_value)

        result.parents.append((A, _grad_a))

    return result


def ln(A: Tensor):
    result = Tensor(np.log(A.arr))
    result.requires_grad = A.requires_grad

    if A.requires_grad:

        def _grad_a(_value):
            return _value * (1.0 / A.arr)

        result.parents.append((A, _grad_a))

    return result


def log2(A: Tensor):
    result = Tensor(np.log2(A.arr))
    result.requires_grad = A.requires_grad

    if A.requires_grad:

        def _grad_a(_value):
            return _value * (1.0 / (A.arr * np.log(2.0)))

        result.parents.append((A, _grad_a))

    return result


def MSE(x_1: Tensor, x_2: Tensor):
    assert x_1.shape == x_2.shape

    n = x_1.arr.size  # Get the number of elements (for averaging)
    diff = x_1 - x_2
    result = (diff**2).sum_np() / n  # Divide by n here

    if x_1.requires_grad:

        def _grad_x1(incoming_grad):
            local_grad = (2.0 / n) * (x_1.arr - x_2.arr)
            sum_axes = get_axes_broadcasting(incoming_grad, x_1)
            return np.sum(
                incoming_grad * local_grad, axis=tuple(sum_axes), keepdims=True
            )

        result.parents.append((x_1, _grad_x1))

    if x_2.requires_grad:

        def _grad_x2(incoming_grad):
            local_grad = (2.0 / n) * (x_2.arr - x_1.arr)
            sum_axes = get_axes_broadcasting(incoming_grad, x_2)
            return np.sum(
                incoming_grad * local_grad, axis=tuple(sum_axes), keepdims=True
            )

        result.parents.append((x_2, _grad_x2))

    return result


def BCE(x: Tensor, y: Tensor) -> Tensor:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    n = x.arr.size  # Total number of elements (for averaging)
    epsilon = 1e-15  # Small value for clipping

    x_clipped = np.clip(x.arr, epsilon, 1.0 - epsilon)

    loss_val = -(y.arr * np.log(x_clipped) + (1 - y.arr) * np.log(1 - x_clipped))
    result = Tensor(
        np.sum(loss_val) / n, requires_grad=(x.requires_grad or y.requires_grad)
    )

    if x.requires_grad:

        def _grad_x(incoming_grad):
            local_grad = (x_clipped - y.arr) / (x_clipped * (1.0 - x_clipped))
            sum_axes = get_axes_broadcasting(incoming_grad, x)
            return np.sum(
                incoming_grad * local_grad, axis=tuple(sum_axes), keepdims=True
            )

        result.parents.append((x, _grad_x))
        # x.children.append(result)

    return result


def sigmoid(A: Tensor):
    A = _to_var(A)
    result = Tensor(1.0 / (1.0 + np.exp(-A.arr)))
    result.requires_grad = A.requires_grad

    if result.requires_grad:

        def _grad_a(_value):
            sig = 1.0 / (1.0 + np.exp(-A.arr))
            local_grad = sig * (1 - sig)
            return _value * local_grad

        result.parents.append((A, _grad_a))
    return result


def relu(A: Tensor):
    A = _to_var(A)
    result = Tensor(np.maximum(0, A.arr))
    result.requires_grad = A.requires_grad

    if result.requires_grad:

        def _grad_relu(_value):
            local_grad = np.array((A.arr > 0)).astype(A.arr.dtype)
            return _value * local_grad

        result.parents.append((A, _grad_relu))

    return result


def softmax(x: Tensor) -> Tensor:
    x = _to_var(x)

    # --- Forward Pass ---
    x_shifted = x.arr - np.max(x.arr, axis=-1, keepdims=True)  # Subtract max for stability
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    result_arr = exp_x / sum_exp_x
    result = Tensor(result_arr, requires_grad=x.requires_grad)

    if x.requires_grad:
        def _grad_x(incoming_grad):
            # Combine "subtract max" and gradient calculation
            # This is the key optimization
            return (incoming_grad - np.sum(incoming_grad * result.arr, axis=-1, keepdims=True)) * result.arr

        result.parents.append((x, _grad_x))

    return result
