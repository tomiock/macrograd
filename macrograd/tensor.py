from __future__ import annotations  # do not touch

from collections import defaultdict
from typing import Hashable, Optional

import math
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
            self.node = self.graph.nodes[self.node_id]
        elif array is not None:
            self.node_id = self.graph.add_node(
                op=Ops.CONST, input_ids=(), static_data=array
            )
            self.node = self.graph.nodes[self.node_id]
        else:
            raise TypeError("must provide data for const or node_id from operation")

    @property
    def grad(self):
        return self._grad

    def __add__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        result_id = self.graph.add_node(Ops.ADD, (self.node_id, other.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad),
            _node_id=result_id,
        )
        return result

    def __mul__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        result_id = self.graph.add_node(Ops.MUL, (self.node_id, other.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad),
            _node_id=result_id,
        )
        return result

    def __pow__(self, exp: TensorLike) -> Tensor:
        if not isinstance(exp, Tensor):
            exp = Tensor(array=exp)
        result_id = self.graph.add_node(Ops.POW, (self.node_id, exp.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or exp.requires_grad),
            _node_id=result_id,
        )
        return result

    def __matmul__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        result_id = self.graph.add_node(Ops.MATMUL, (self.node_id, other.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad),
            _node_id=result_id,
        )
        return result

    @property
    def T(self) -> Tensor:
        result_id = self.graph.add_node(Ops.TRANSPOSE, (self.node_id,))
        result = Tensor(
            requires_grad=self.requires_grad,
            _node_id=result_id,
        )
        return result

    def transpose(self, axes: Optional[tuple | list] = None) -> Tensor:
        kwargs = dict()
        if axes is not None:
            kwargs["axes"] = axes
        result_id = self.graph.add_node(Ops.TRANSPOSE, (self.node_id,), kwargs=kwargs)
        result = Tensor(
            requires_grad=self.requires_grad,
            _node_id=result_id,
        )
        return result

    def sum(self, axis=None, keepdims=False) -> Tensor:
        kwargs = dict()
        kwargs["axis"] = axis
        kwargs["keepdims"] = keepdims

        result_id = self.graph.add_node(Ops.SUM, (self.node_id,), kwargs=kwargs)
        result = Tensor(requires_grad=self.requires_grad, _node_id=result_id)
        return result

    def exp(self) -> Tensor:
        result_id = self.graph.add_node(Ops.EXP, (self.node_id,))
        result = Tensor(requires_grad=self.requires_grad, _node_id=result_id)
        return result

    def log(self, base: float | str = "e"):
        graph = get_default_graph()
        result_id = graph.add_node(
            op=Ops.LOG, input_ids=(self.node_id,), kwargs={"base": base}
        )
        result = Tensor(requires_grad=self.requires_grad, _node_id=result_id)
        return result

    def reshape(self, shape: int | tuple):
        result_id = self.graph.add_node(
            Ops.RESHAPE, (self.node_id,), shape_reshape=shape
        )
        result = Tensor(requires_grad=self.requires_grad, _node_id=result_id)
        return result

    def _backward(self):
        raise NotImplementedError

    def backprop(self):
        self._backward()
        self.parents = set()

    def __repr__(self) -> str:
        return f"{self.data}, grad={self.grad}, shape={self.shape}, rgrad={self.requires_grad}"

    def sqrt(self) -> Tensor:
        return self**0.5

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
