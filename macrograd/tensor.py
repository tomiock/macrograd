from collections.abc import Callable
import numpy as np
import numpy.typing as npt


def _to_var(x):
    if isinstance(x, Tensor):
        x.arr = np.array(x.arr)
        return x
    else:
        x_var = Tensor(x)
        x_var.arr = np.array(x_var.arr)
        return x_var


type TensorLike = Tensor | int | float | np.ndarray


class Tensor:
    def __init__(self, array: npt.ArrayLike, requires_grad=False, precision=None):
        self.requires_grad = requires_grad
        self._grad: npt.ArrayLike | None = None
        self.parents: list[
            tuple["Tensor", Callable]
        ] = []  # Use string for forward reference
        self.precision = precision
        self.arr = np.array(array, dtype=precision)
        self.dim = self.arr.ndim
        self.shape = self.arr.shape

    @property
    def grad(self):
        return self._grad

    def zero_grad(self):
        self._grad = np.zeros_like(self.arr)

    def backprop(self):
        self._backward()
        self.parents = []

    def __repr__(self) -> str:
        return f"{self.arr}, grad={self.grad}, shape={self.shape}"

    def reshape(self, *args):
        return Tensor(self.arr.reshape(args), requires_grad=self.requires_grad)

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

        _value = np.array(_value, dtype=self.precision)
        self._grad = _value

        del visited
        del visit_stack
        
        for node in reversed(stack):
            if node.grad is None:
                continue
            for prev_node, local_grad in node.parents:
                if prev_node.requires_grad:
                    if prev_node._grad is None:
                        prev_node._grad = np.zeros(prev_node.shape)
                    grad_delta = local_grad(node._grad)
                    if grad_delta.shape != prev_node._grad.shape:
                        grad_delta = grad_delta.reshape(prev_node._grad.shape)
                    prev_node._grad += grad_delta
            node.parents = []

    def __add__(self, other):
        return vAdd(self, other)

    def __radd__(self, other):
        return vAdd(other, self)

    def __sub__(self, other):
        return vAdd(self, -other)

    def __rsub__(self, other):
        return vAdd(other, -self)

    def __mul__(self, other):
        return vMul(self, other)

    def __rmul__(self, other):
        return vMul(other, self)

    def __truediv__(self, other):
        return vMul(self, vPow(other, -1.0))

    def __rtruediv__(self, other):
        return vMul(other, vPow(self, -1.0))

    def __matmul__(self, other):
        return vMatMul(self, other)

    def __rmatmul__(self, other):
        return vMatMul(other, self)

    def __neg__(self):
        return vMul(self, -1.0)

    def __pow__(self, exponent):
        return vPow(self, exponent)

    @property
    def T(self):
        return vTranspose(self)

    def sqrt(self):
        return vSqrt(self)

    def sum(self, axis=None, keepdims=False):
        result_value = np.sum(self.arr, axis=axis, keepdims=keepdims)
        result = Tensor(result_value, requires_grad=self.requires_grad)

        if self.requires_grad:

            def _grad_sum(_value):
                if axis is None:
                    return _value * np.ones_like(self.arr)

                if not keepdims:
                    # We need to "un-squeeze" the _value array to the same number
                    # of dimensions as self.arr before broadcasting.
                    _value_expanded = np.expand_dims(_value, axis=axis)
                    return _value_expanded * np.ones_like(self.arr)
                else:
                    return _value * np.ones_like(self.arr)

            result.parents.append((self, _grad_sum))

        return result


def get_axes_broadcasting(incoming_grad, tensor: Tensor):
    sum_axes = []
    for i in range(len(incoming_grad.shape)):
        if i < len(tensor.shape):
            if tensor.shape[i] == 1 and incoming_grad.shape[i] > 1:
                sum_axes.append(i)
        elif i >= len(tensor.shape):
            sum_axes.append(i)
    return sum_axes


def vSqrt(A: Tensor):
    A = _to_var(A)
    result = Tensor(np.sqrt(A.arr), requires_grad=A.requires_grad)
    if A.requires_grad:

        def _grad_sqrt(_value):
            return _value * (0.5 / np.sqrt(A.arr))

        result.parents.append((A, _grad_sqrt))
    return result


def vTranspose(A: Tensor):
    A = _to_var(A)
    result = Tensor(A.arr.T, requires_grad=A.requires_grad)
    if A.requires_grad:

        def _grad_t(_value):
            return _value.T

        result.parents.append((A, _grad_t))
    return result


def vMatMul(A: Tensor, B: Tensor):
    A = _to_var(A)
    B = _to_var(B)
    result = Tensor(
        np.dot(A.arr, B.arr), requires_grad=(A.requires_grad or B.requires_grad)
    )
    if A.requires_grad:

        def _grad_a(_value):
            return np.matmul(_value, B.arr.T)

        result.parents.append((A, _grad_a))
    if B.requires_grad:

        def _grad_b(_value):
            return np.matmul(A.arr.T, _value)

        result.parents.append((B, _grad_b))
    return result


def vAdd(A: Tensor | float | int, B: Tensor | float | int):
    A = _to_var(A)
    B = _to_var(B)
    result = Tensor(A.arr + B.arr, requires_grad=(A.requires_grad or B.requires_grad))
    if A.requires_grad:

        def _grad_a(_value):
            sum_axes = get_axes_broadcasting(_value, A)
            return np.sum(_value, axis=tuple(sum_axes), keepdims=True)

        result.parents.append((A, _grad_a))
    if B.requires_grad:

        def _grad_b(_value):
            sum_axes = get_axes_broadcasting(_value, B)
            return np.sum(_value, axis=tuple(sum_axes), keepdims=True)

        result.parents.append((B, _grad_b))
    return result


def vMul(A: TensorLike, B: TensorLike):
    A = _to_var(A)
    B = _to_var(B)
    result = Tensor(A.arr * B.arr, requires_grad=(A.requires_grad or B.requires_grad))
    if A.requires_grad:

        def _grad_a(_value):
            sum_axes = get_axes_broadcasting(_value, A)
            return np.sum(_value * B.arr, axis=tuple(sum_axes), keepdims=True)

        result.parents.append((A, _grad_a))

    if B.requires_grad:

        def _grad_b(incoming_grad):
            sum_axes = get_axes_broadcasting(incoming_grad, B)
            return np.sum(incoming_grad * A.arr, axis=tuple(sum_axes), keepdims=True)

        result.parents.append((B, _grad_b))
    return result


def vPow(A: TensorLike, exponent: TensorLike) -> Tensor:
    A = _to_var(A)
    exponent = _to_var(exponent)  # Corrected: Use _to_var
    result = Tensor(
        np.power(A.arr, exponent.arr),
        requires_grad=(A.requires_grad or exponent.requires_grad),
    )
    if A.requires_grad:

        def _grad_a(_value):
            local_grad = exponent.arr * np.power(A.arr, exponent.arr - 1)
            sum_axes = get_axes_broadcasting(_value, A)
            return np.sum(_value * local_grad, axis=tuple(sum_axes), keepdims=True)

        result.parents.append((A, _grad_a))

    if exponent.requires_grad:

        def _grad_exponent(_value):
            local_grad = np.power(A.arr, exponent.arr) * np.log(A.arr)
            sum_axes = get_axes_broadcasting(_value, exponent)
            return np.sum(_value * local_grad, axis=tuple(sum_axes), keepdims=True)

        result.parents.append((exponent, _grad_exponent))
    return result
