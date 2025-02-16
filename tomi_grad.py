from collections.abc import Callable
import numpy as np
import numpy.typing as npt


class Var:
    def __init__(self, array: npt.ArrayLike, requires_grad=True, precision=np.float64):
        self.requires_grad = requires_grad
        self._grad: np.ndarray | None = None
        self.pointers: list[tuple[Var, Callable]] = []

        # TODO: type check this
        self.precision = precision

        self.arr = np.array(array, dtype=precision)

        self.dim = self.arr.ndim
        self.shape = self.arr.shape

    def backprop(self):
        self._backward()

    @property
    def grad(self):
        return self._grad

    def zeroGrad(self):
        self._grad = np.zeros_like(self.arr)
        for var, _ in self.pointers:
            var.zeroGrad()

    def _backward(self, _value: np.ndarray | float = 1.0):
        if not self.requires_grad:
            return

        if self._grad is None:
            self._grad = np.zeros_like(self.arr)
        _value = np.array(_value, dtype=self.precision)

        for _var, _local_grad in self.pointers:
            if _var.requires_grad:
                if callable(_local_grad):
                    _var._backward(_local_grad(_value))
                else:
                    _var._backward(_value * _local_grad)

        self._grad += _value

    def sum(self):
        result_value = np.sum(self.arr)
        result = Var(result_value, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _grad_sum(_value):
                return _value * np.ones_like(self.arr)

            result.pointers.append((self, _grad_sum))

        return result

    def _forward(self):
        raise NotImplementedError

    def get_operation(self):
        raise NotImplementedError

    def get_consumers(self):
        raise NotImplementedError

    def get_inputs(self):
        raise NotImplementedError

    def __add__(self, other):
        other = _to_var(other)
        return vAdd(self, other)

    def __radd__(self, other):
        other = _to_var(other)
        return vAdd(other, self)

    def __sub__(self, other):
        other = _to_var(other)
        return vSub(self, other)

    def __rsub__(self, other):
        other = _to_var(other)
        return vSub(other, self)

    def __mul__(self, other):
        other = _to_var(other)
        return vMul(self, other)

    def __rmul__(self, other):
        other = _to_var(other)
        return vMul(other, self)

    def __pow__(self, other):
        other = _to_var(other)
        # return vPow(self, other)
        raise NotImplementedError

    def __matmul__(self, other):
        other = _to_var(other)
        return vMatMul(self, other)

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return str(self.arr)


def _to_var(x):
    if isinstance(x, Var):
        x.arr = np.array(x.arr)
        return x
    else:
        x_var = Var(x)
        x_var.arr = np.array(x_var.arr)
        return x_var


def vMatMul(A: Var, B: Var):
    A = _to_var(A)
    B = _to_var(B)

    result = np.matmul(A.arr, B.arr)
    required_grad = A.requires_grad or B.requires_grad
    result = Var(result, requires_grad=required_grad)

    if A.requires_grad:

        def _grad_a(input_grad):
            return np.matmul(input_grad, B.arr.T)  # G @ B.T

        result.pointers.append((A, _grad_a))

    if B.requires_grad:

        def _grad_b(input_grad):
            return np.matmul(A.arr.T, input_grad)  # A.T @ G

        result.pointers.append((B, _grad_b))

    return result


def vAdd(A: Var | float | int, B: Var | float | int):
    A = _to_var(A)
    B = _to_var(B)

    result = Var(A.arr + B.arr)

    if A.requires_grad:
        result.requires_grad = True

        def _grad_a(_value):
            return _value

        result.pointers.append((A, _grad_a))

    if B.requires_grad:
        result.requires_grad = True

        def _grad_b(_value):
            return _value

        result.pointers.append((B, _grad_b))

    return result


def vSub(A: Var | float | int, B: Var | float | int):
    A = _to_var(A)
    B = _to_var(B)

    result = Var(A.arr - B.arr)

    result.pointers.append((A, np.array(-1)))
    result.pointers.append((B, np.array(-1)))

    return result


def vMul(A: Var, B: Var):
    A = _to_var(A)
    B = _to_var(B)

    result = Var(A.arr * B.arr)

    result.pointers.append((A, B.arr))
    result.pointers.append((B, A.arr))

    return result


"""
def vPow(A: Var, exponent: Var):
    A = _to_var(A)
    exponent = _to_var(exponent)

    _array = np.array(A.arr)
    _exponent = np.array(exponent.arr)
    result = Var(np.power(_array, _exponent))

    if A.requires_grad:
        result._grad = 0.0
        dev = exponent * np.power(A.arr, _exponent - 1)
        result.pointers.append((A, dev))

    return result
"""
