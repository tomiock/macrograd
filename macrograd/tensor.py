from __future__ import annotations  # do not touch

from collections import defaultdict
from typing import Hashable, Optional

import numpy as np
import numpy.typing as npt

from macrograd.engine import get_default_graph, Ops

type ArrayLike = int | float | np.ndarray | list | tuple
type TensorLike = Tensor | int | float | np.ndarray | list


def _to_var(x: TensorLike) -> "Tensor":
    if isinstance(x, Tensor):
        return x
    else:
        return Tensor(np.array(x))


class Tensor:
    def __init__(
        self,
        array: Optional[ArrayLike] = None,
        requires_grad=False,
        precision=np.float32,
        _node_id: Optional[Hashable] = None,
    ):
        self.graph = get_default_graph()
        self.requires_grad = requires_grad
        self.data = np.array(array, dtype=precision)
        self.shape = self.data.shape
        self._grad: npt.ArrayLike | None = None
        self.node_id: Hashable

        if _node_id is not None:
            self.node_id = _node_id
        elif array is not None:
            self.node_id = self.graph.add_node(
                op=Ops.CONST, input_ids=(), static_data=array
            )
        else:
            raise TypeError("must provide data for const or node_id from operation")

    @property
    def grad(self):
        return self._grad

    # DONE
    def __add__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        result_id = self.graph.add_node(Ops.ADD, (self.node_id, other.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad),
            _node_id=result_id,
        )
        return result

    # DONE
    def __mul__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        result_id = self.graph.add_node(Ops.MUL, (self.node_id, other.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad),
            _node_id=result_id,
        )
        return result

    # DONE
    def __pow__(self, exp: TensorLike) -> Tensor:
        if not isinstance(exp, Tensor):
            exp = Tensor(array=exp)
        result_id = self.graph.add_node(Ops.POW, (self.node_id, exp.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or exp.requires_grad),
            _node_id=result_id,
        )
        return result

    # DONE
    def __matmul__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        result_id = self.graph.add_node(Ops.MATMUL, (self.node_id, other.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad),
            _node_id=result_id,
        )
        return result

    # DONE
    @property
    def T(self) -> Tensor:
        result_id = self.graph.add_node(Ops.TRANSPOSE, (self.node_id,))
        result = Tensor(
            requires_grad=self.requires_grad,
            _node_id=result_id,
        )
        return result

    # DONE
    def sum(self, axis=None, keepdims=False) -> Tensor:
        kwargs = dict()
        kwargs['axis'] = axis
        kwargs['keepdims'] = keepdims

        result_id = self.graph.add_node(Ops.SUM, (self.node_id,), kwargs=kwargs)
        result = Tensor(requires_grad=self.requires_grad, _node_id=result_id)
        return result

    # DONE
    def reshape(self, *args):
        result_id = self.graph.add_node(Ops.RESHAPE, (self.node_id,))
        result = Tensor(requires_grad=self.requires_grad, _node_id=result_id)
        return result

    def zero_grad(self):
        self._grad = tensor_zeros_like(self.data)

    def backprop(self):
        self._backward()
        self.parents = set()

    def __repr__(self) -> str:
        return f"{self.data}, grad={self.grad}, shape={self.shape}, rgrad={self.requires_grad}"

    def sqrt(self) -> Tensor:
        return self ** 0.5

    def __radd__(self, other: TensorLike) -> Tensor:
        return _to_var(other) + self

    def __sub__(self, other: TensorLike) -> Tensor:
        return self + (-_to_var(other))

    def __rsub__(self, other: TensorLike) -> Tensor:
        return _to_var(other) + (-self)

    def __rmul__(self, other: TensorLike) -> Tensor:
        return _to_var(other) * self

    def __truediv__(self, other: TensorLike) -> Tensor:
        return self * (_to_var(other) ** np.array(-1.0))

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        return _to_var(other) * (self ** np.array(-1.0))

    def __rmatmul__(self, other: TensorLike) -> Tensor:
        return _to_var(other) @ self

    def __neg__(self) -> Tensor:
        return self * -1.0


def get_axes_broadcasting(_data: np.ndarray, arr: np.ndarray) -> list[int]:
    sum_axes = []
    for i in range(len(_data.shape)):
        if i < len(arr.shape):
            if arr.shape[i] == 1 and _data.shape[i] > 1:
                sum_axes.append(i)
        elif i >= len(arr.shape):
            sum_axes.append(i)
    return sum_axes


# --- Gradient Functions (External, NumPy-based) ---


def grad_add(_data: np.ndarray, arr1: np.ndarray) -> np.ndarray:
    sum_axes = get_axes_broadcasting(_data, arr1)
    return tensor_sum(_data, axis=tuple(sum_axes), keepdims=True)


def grad_mul(_data: np.ndarray, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    sum_axes = get_axes_broadcasting(_data, arr1)
    return tensor_sum(tensor_mul(_data, arr2), axis=tuple(sum_axes), keepdims=True)


def grad_matmul(
    _data: np.ndarray,
    arr1: np.ndarray,
    arr2: np.ndarray,
    is_a: bool,
) -> np.ndarray:
    if is_a:
        return tensor_matmul(_data, tensor_transpose(arr2))
    else:
        return tensor_matmul(tensor_transpose(arr1), _data)


def grad_pow(
    _data: np.ndarray,
    arr: np.ndarray,
    exponent: np.ndarray,
    is_a: bool,
) -> np.ndarray:
    if is_a:
        local_grad = tensor_mul(exponent, tensor_pow(arr, np.array(exponent - 1)))
        sum_axes = get_axes_broadcasting(_data, arr)
        return tensor_sum(
            tensor_mul(_data, local_grad), axis=tuple(sum_axes), keepdims=True
        )
    else:
        local_grad = tensor_mul(tensor_pow(arr, exponent), tensor_log(arr))
        sum_axes = get_axes_broadcasting(_data, exponent)
        return tensor_sum(
            tensor_mul(_data, local_grad), axis=tuple(sum_axes), keepdims=True
        )


def grad_sqrt(_data: np.ndarray, arr: np.ndarray) -> np.ndarray:
    return tensor_mul(_data, (0.5 / tensor_sqrt(arr)))


def grad_sum(_data: np.ndarray, arr: np.ndarray, axis, keepdims) -> np.ndarray:
    if axis is None:
        return tensor_mul(_data, tensor_ones_like(arr))

    if not keepdims:
        _data_expanded = tensor_expand_dims(_data, axis=axis)
        return tensor_mul(_data_expanded, tensor_ones_like(arr))
    else:
        return tensor_mul(_data, tensor_ones_like(arr))


def grad_transpose(_data: np.ndarray) -> np.ndarray:
    return _data.T


# --- NumPy-based functions ---
def tensor_sqrt(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(arr)


def tensor_transpose(arr: np.ndarray) -> np.ndarray:
    return arr.T


def tensor_matmul(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.dot(arr1, arr2)


def tensor_add(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return arr1 + arr2


def tensor_mul(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return arr1 * arr2


def tensor_pow(arr: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    return np.power(arr, exponent)


def tensor_sum(arr: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    return np.sum(arr, axis=axis, keepdims=keepdims)


def tensor_reshape(arr: np.ndarray, shape) -> np.ndarray:
    return arr.reshape(shape)


def tensor_zeros_like(arr: np.ndarray) -> np.ndarray:
    return np.zeros_like(arr)


def tensor_zeros(shape: tuple) -> np.ndarray:
    return np.zeros(shape)


def tensor_ones_like(arr: np.ndarray) -> np.ndarray:
    return np.ones_like(arr)


def tensor_expand_dims(arr: np.ndarray, axis) -> np.ndarray:
    return np.expand_dims(arr, axis)


def tensor_log(arr: np.ndarray) -> np.ndarray:
    return np.log(arr)
