from collections.abc import Callable
import numpy as np
import numpy.typing as npt

type ArrayLike = int | float | np.ndarray | list | tuple
type TensorLike = Tensor | int | float | np.ndarray | list


def _to_var(x: TensorLike) -> "Tensor":
    if isinstance(x, Tensor):
        return x
    else:
        return Tensor(np.array(x))


class Tensor:
    def __init__(self, array: ArrayLike, requires_grad=False, precision=np.float32):
        self.requires_grad = requires_grad
        self._grad: npt.ArrayLike | None = None
        self.parents: list[
            tuple[Tensor, Callable]
        ] = []  # Use string for forward reference
        self.data = np.array(array, dtype=precision)
        self.dim = self.data.ndim
        self.shape = self.data.shape

    @property
    def grad(self):
        return self._grad

    def zero_grad(self):
        self._grad = tensor_zeros_like(self.data)

    def backprop(self):
        self._backward()
        self.parents = []

    def __repr__(self) -> str:
        return f"{self.data}, grad={self.grad}, shape={self.shape}"

    def reshape(self, *args):
        return Tensor(
            tensor_reshape(self.data, args), requires_grad=self.requires_grad
        )

    def _backward(self, _value: np.ndarray = np.array(1.0)):
        if not self.requires_grad:
            return
        visited = set()
        visit_stack = [self]
        stack = []

        while visit_stack:
            node = visit_stack[-1]
            all_parents_visited = True
            for p_node, _ in node.parents:
                if p_node not in visited:
                    all_parents_visited = False
                    visit_stack.append(p_node)
                    break

            if all_parents_visited:
                visit_stack.pop()
                if node not in visited:
                    visited.add(node)
                    stack.append(node)

        self._grad = _value

        del visited
        del visit_stack

        for node in reversed(stack):
            if node.grad is None:
                continue
            for prev_node, local_grad_fn in node.parents:
                if prev_node.requires_grad:
                    if prev_node._grad is None:
                        prev_node._grad = tensor_zeros(prev_node.shape)
                    # Get the local gradient.  Critically, pass *arrays*, not Tensors.
                    grad_delta = local_grad_fn(node._grad)
                    if grad_delta.shape != prev_node._grad.shape:
                        grad_delta = tensor_reshape(grad_delta, prev_node._grad.shape)
                    prev_node._grad += grad_delta
            node.parents = []

    def __add__(self, other: TensorLike) -> "Tensor":
        other = _to_var(other)
        result = Tensor(
            tensor_add(self.data, other.data),
            requires_grad=(self.requires_grad or other.requires_grad),
        )
        if self.requires_grad:
            # Pass arrays to the external gradient function.
            result.parents.append(
                (self, lambda x: grad_add(x, self.data, other.data))
            )  # Pass A and B
        if other.requires_grad:
            result.parents.append(
                (other, lambda x: grad_add(x, other.data, self.data))
            )  # Pass B and A (order matters for broadcasting)
        return result

    def __radd__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) + self

    def __sub__(self, other: TensorLike) -> "Tensor":
        return self + (-_to_var(other))

    def __rsub__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) + (-self)

    def __mul__(self, other: TensorLike) -> "Tensor":
        other = _to_var(other)
        result = Tensor(
            tensor_mul(self.data, other.data),
            requires_grad=(self.requires_grad or other.requires_grad),
        )
        if self.requires_grad:
            result.parents.append(
                (self, lambda x: grad_mul(x, self.data, other.data))
            )
        if other.requires_grad:
            result.parents.append(
                (other, lambda x: grad_mul(x, other.data, self.data))
            )
        return result

    def __rmul__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) * self

    def __truediv__(self, other: TensorLike) -> "Tensor":
        return self * (_to_var(other) ** -1.0)

    def __rtruediv__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) * (self**-1.0)

    def __matmul__(self, other: TensorLike) -> "Tensor":
        other = _to_var(other)
        result = Tensor(
            tensor_matmul(self.data, other.data),
            requires_grad=(self.requires_grad or other.requires_grad),
        )
        if self.requires_grad:
            result.parents.append(
                (self, lambda x: grad_matmul(x, self.data, other.data, is_a=True))
            )
        if other.requires_grad:
            result.parents.append(
                (other, lambda x: grad_matmul(x, self.data, other.data, is_a=False))
            )
        return result

    def __rmatmul__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) @ self

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __pow__(self, exponent: TensorLike) -> "Tensor":
        exponent = _to_var(exponent)
        result = Tensor(
            tensor_pow(self.data, exponent.data),
            requires_grad=(self.requires_grad or exponent.requires_grad),
        )
        if self.requires_grad:
            result.parents.append(
                (self, lambda x: grad_pow(x, self.data, exponent.data, is_a=True))
            )
        if exponent.requires_grad:
            result.parents.append(
                (
                    exponent,
                    lambda x: grad_pow(x, self.data, exponent.data, is_a=False),
                )
            )
        return result

    @property
    def T(self) -> "Tensor":
        result = Tensor(tensor_transpose(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.parents.append((self, lambda x: grad_transpose(x)))
        return result

    def sqrt(self) -> "Tensor":
        result = Tensor(tensor_sqrt(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.parents.append((self, lambda x: grad_sqrt(x, self.data)))
        return result

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        result_value = tensor_sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_value, requires_grad=self.requires_grad)

        if self.requires_grad:

            def _grad_sum(_data: np.ndarray) -> np.ndarray:
                return grad_sum(_data, self.data, axis, keepdims)

            result.parents.append((self, _grad_sum))

        return result


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


def grad_add(_data: np.ndarray, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
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
        local_grad = tensor_mul(exponent, tensor_pow(arr, exponent - 1))
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


def tensor_zeros(shape) -> np.ndarray:
    return np.zeros(shape)


def tensor_ones_like(arr: np.ndarray) -> np.ndarray:
    return np.ones_like(arr)


def tensor_expand_dims(arr: np.ndarray, axis) -> np.ndarray:
    return np.expand_dims(arr, axis)


def tensor_log(arr: np.ndarray) -> np.ndarray:
    return np.log(arr)

