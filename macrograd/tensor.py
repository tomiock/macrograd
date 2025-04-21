from __future__ import annotations  # do not touch

from typing import Hashable, Optional

import numpy as np
import numpy.typing as npt

from macrograd.engine import get_default_graph, Ops, Graph, executor
from macrograd.backward import _backward

type ArrayLike = int | float | np.ndarray | list
type TensorLike = Tensor | int | float | np.ndarray | list


class Tensor:
    def __init__(
        self,
        array: Optional[ArrayLike] = None,
        requires_grad=False,
        graph: Optional[Graph] = None,
        _node_id: Optional[Hashable] = None,
    ):
        self.node_id: Hashable

        if graph is None:
            self.graph = get_default_graph()
        else:
            self.graph = graph

        self.requires_grad = requires_grad

        # handle input given
        if array is not None:
            if isinstance(array, np.ndarray):
                self._data = array
            elif isinstance(array, (list, int, float)):
                self._data = np.array(array)
            else:
                raise TypeError(
                    f"Input to tensor must be list, int, float or np.ndarray, got {type(array)}"
                )
            self.shape = self._data.shape

            # create a CONST NODE
            self.node_id = self.graph.add_node(
                op=Ops.CONST, input_ids=(), static_data=array, rg=self.requires_grad
            )
            self.node = self.graph.nodes[self.node_id]
        else:
            if _node_id is not None:
                self.node_id = _node_id
                self.node = self.graph.nodes[self.node_id]
            else:
                raise TypeError("must provide data for const or node_id from operation")

    @property
    def grad(self):
        return self.node.grad

    @property
    def data(self):
        self._data = self.node.computed_tensor
        return self._data

    def realize(self):
        if self.node.op == Ops.CONST:
            return self.data
        else:
            executor(self.graph)
            self._data = self.node.computed_tensor

    def backprop(self):
        _backward(self.graph, self.node_id)

    def __add__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other, graph=self.graph)
        rg = self.requires_grad or other.requires_grad
        result_id = self.graph.add_node(Ops.ADD, (self.node_id, other.node_id), rg=rg)
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def __mul__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other, graph=self.graph)
        rg = self.requires_grad or other.requires_grad
        result_id = self.graph.add_node(Ops.MUL, (self.node_id, other.node_id), rg=rg)
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def __pow__(self, exp: TensorLike) -> Tensor:
        if not isinstance(exp, Tensor):
            exp = Tensor(array=exp, graph=self.graph)

        rg = self.requires_grad or exp.requires_grad
        result_id = self.graph.add_node(Ops.POW, (self.node_id, exp.node_id), rg=rg)
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def __matmul__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other, graph=self.graph)

        rg = self.requires_grad or other.requires_grad
        result_id = self.graph.add_node(
            Ops.MATMUL, (self.node_id, other.node_id), rg=rg
        )
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    @property
    def T(self) -> Tensor:
        result = self.transpose()
        return result

    def transpose(self, axes: Optional[tuple | list] = None) -> Tensor:
        kwargs = dict()
        if axes is not None:
            kwargs["axes"] = axes
        rg = self.requires_grad
        result_id = self.graph.add_node(
            Ops.TRANSPOSE, (self.node_id,), kwargs=kwargs, rg=rg
        )
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def max(self, axis=None, keepdims=False) -> Tensor:
        kwargs = dict()
        kwargs["axis"] = axis
        kwargs["keepdims"] = keepdims
        rg = self.requires_grad

        result_id = self.graph.add_node(Ops.MAX, (self.node_id,), kwargs=kwargs, rg=rg)
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def sum(self, axis=None, keepdims=False) -> Tensor:
        kwargs = dict()
        kwargs["axis"] = axis
        kwargs["keepdims"] = keepdims
        rg = self.requires_grad

        result_id = self.graph.add_node(Ops.SUM, (self.node_id,), kwargs=kwargs, rg=rg)
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def exp(self) -> Tensor:
        rg = self.requires_grad
        result_id = self.graph.add_node(Ops.EXP, (self.node_id,), rg=rg)
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def log(self, base: float | str = "e"):
        rg = self.requires_grad
        result_id = self.graph.add_node(
            op=Ops.LOG, input_ids=(self.node_id,), kwargs={"base": base}, rg=rg
        )
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def reshape(self, shape: int | tuple):
        rg = self.requires_grad
        result_id = self.graph.add_node(
            Ops.RESHAPE,
            (self.node_id,),
            kwargs={"shape": shape},
            rg=rg,
        )
        result = Tensor(
            requires_grad=rg,
            _node_id=result_id,
            graph=self.graph,
        )
        return result

    def relu(self) -> Tensor:
        return relu(self)

    def __repr__(self) -> str:
        return f"{self.node.computed_tensor}, grad={self.grad}, shape={self.node.shape}, rgrad={self.requires_grad}"

    def sqrt(self) -> Tensor:
        return self**0.5

    def __radd__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        return other + self

    def __sub__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        return self + -(other)

    def __rsub__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        return other + (-self)

    def __rmul__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        return other * self

    def __truediv__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        return self * (other ** np.array(-1.0))

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        return other * (self ** np.array(-1.0))

    def __rmatmul__(self, other: TensorLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        return other @ self

    def __neg__(self) -> Tensor:
        return self * -1.0


def relu(in_tensor: Tensor) -> Tensor:
    graph = in_tensor.graph
    rg = in_tensor.requires_grad
    node_id = graph.add_node(
        op=Ops.RELU,
        input_ids=(in_tensor.node_id,),
        rg=rg,
    )
    return Tensor(requires_grad=rg, _node_id=node_id, graph=graph)
