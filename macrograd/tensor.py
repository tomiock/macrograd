from __future__ import annotations  # do not touch


from collections import defaultdict
from dataclasses import dataclass, field
import graphviz
from collections.abc import Callable
from enum import Enum, IntEnum, auto
from typing import Any, Hashable, Optional
import numpy as np
import numpy.typing as npt
import uuid

type ArrayLike = int | float | np.ndarray | list | tuple
type TensorLike = Tensor | int | float | np.ndarray | list


def _to_var(x: TensorLike) -> "Tensor":
    if isinstance(x, Tensor):
        return x
    else:
        return Tensor(np.array(x))


class FastEnum(IntEnum):
    def __str__(self):
        return Enum.__str__(self)

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])


class Ops(FastEnum):
    ADD = auto()
    MUL = auto()
    MATMUL = auto()
    SUM = auto()
    SQRT = auto()
    POW = auto()
    CONV2D = auto()
    CONV1D = auto()
    CONST = auto()


@dataclass
class Node:
    id: Hashable
    op: Optional[Ops] = None
    inputs: tuple[Hashable, ...] = field(default_factory=tuple)
    succesors: set[Hashable] = field(default_factory=set)
    data: Any = None
    device: str = "cpu"
    shape: Optional[tuple[int, ...]] = None
    dtype: Optional[np.dtype] = None

    # TODO: handle static_data

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return self.id == other.id
        return NotImplemented

    def __repr__(self) -> str:
        op_name = self.op.name if self.op else "const"
        input_ids = ", ".join(map(repr, self.inputs))
        successor_ids = ", ".join(map(repr, self.succesors))

        return (
            f"{self.__class__.__name__}("
            f"id={self.id!r}, op={op_name}, "
            f"inputs=({input_ids}), "
            f"successors={{{successor_ids}}}, "
            f"data={self.data!r})"
        )


class Graph:
    def __init__(self):
        self.nodes: dict[Hashable, Node] = {}
        self._op_counters: dict[str, int] = defaultdict(int)

    def _get_next_id(self, op_name: str) -> str:
        count = self._op_counters[op_name]
        self._op_counters[op_name] += 1
        new_id = f"{op_name.lower()}_{count}"

        # TODO: do collision check

        return new_id

    def add_node(
        self,
        op: Optional[Ops],
        input_ids: tuple[Hashable, ...],
        static_data: Any = None,
        target_shape: Optional[tuple] = None,
    ) -> Hashable:
        op_name_str = op.name if op else "const"
        new_id = self._get_next_id(op_name_str)

        # TODO: shape inference

        node = Node(id=new_id, op=op, inputs=input_ids, shape=None, dtype=None)
        self.nodes[new_id] = node

        for input_id in input_ids:
            if input_id in self.nodes:
                self.nodes[input_id].succesors.add(new_id)

        return new_id

    def __repr__(self) -> str:
        return " \n".join(map(repr, self.nodes))


_DEFAULT_GRAPH: Optional[Graph] = None


def get_default_graph() -> Graph:
    """Gets the current default graph, creating one if needed."""
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        # debug
        print("[Graph] Initializing default graph.")
        _DEFAULT_GRAPH = Graph()
    return _DEFAULT_GRAPH


def set_default_graph(graph: Optional[Graph]):
    """Allows explicitly setting or clearing the default graph."""
    global _DEFAULT_GRAPH
    _DEFAULT_GRAPH = graph


class Tensor:
    def __init__(
        self,
        array: Optional[ArrayLike] = None,
        requires_grad=False,
        precision=np.float32,
        _node_id: Optional[Hashable] = None,
    ):
        self.node_id = None
        self.graph = get_default_graph()
        self.requires_grad = requires_grad
        self.data = np.array(array, dtype=precision)
        self.shape = self.data.shape
        self._grad: npt.ArrayLike | None = None

        if _node_id is not None:
            self.node_id = _node_id
        elif array is not None:
            self.node_id = self.graph.add_node(op=Ops.CONST, input_ids=())
        else:
            raise TypeError("must provide data for const or node_id from operation")

    @property
    def grad(self):
        return self._grad

    def __add__(self, other: TensorLike) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        result_id = self.graph.add_node(Ops.ADD, (self.node_id, other.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad),
            _node_id=result_id,
        )
        return result

    def __mul__(self, other: TensorLike) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(array=other)
        result_id = self.graph.add_node(Ops.MUL, (self.node_id, other.node_id))
        result = Tensor(
            requires_grad=(self.requires_grad or other.requires_grad),
            _node_id=result_id,
        )
        return result

    def zero_grad(self):
        self._grad = tensor_zeros_like(self.data)

    def backprop(self):
        self._backward()
        self.parents = set()

    def __repr__(self) -> str:
        return f"{self.data}, grad={self.grad}, shape={self.shape}, rgrad={self.requires_grad}"

    def reshape(self, *args):
        return Tensor(tensor_reshape(self.data, args), requires_grad=self.requires_grad)

    def __radd__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) + self

    def __sub__(self, other: TensorLike) -> "Tensor":
        return self + (-_to_var(other))

    def __rsub__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) + (-self)

    def __rmul__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) * self

    def __truediv__(self, other: TensorLike) -> "Tensor":
        return self * (_to_var(other) ** np.array(-1.0))

    def __rtruediv__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) * (self ** np.array(-1.0))

    def __matmul__(self, other: TensorLike) -> "Tensor":
        other = _to_var(other)
        assert type(self.data) is np.ndarray
        assert type(other.data) is np.ndarray
        result = Tensor(
            tensor_matmul(self.data, other.data),
            requires_grad=(self.requires_grad or other.requires_grad),
            _op="matmul",
        )
        if self.requires_grad:

            def _calc_grad_matmul_self(x):
                return grad_matmul(x, self.data, other.data, is_a=True)

            result.parents.add((self, _calc_grad_matmul_self))
        if other.requires_grad:

            def _calc_grad_matmul_other(x):
                return grad_matmul(x, self.data, other.data, is_a=False)

            result.parents.add((other, _calc_grad_matmul_other))
        result._nodes_edges.add(self)
        result._nodes_edges.add(other)
        self._update_node_info(result, other)
        # print(f"{result.shape} = {result._op}({self.label}, {other.label})")
        return result

    def __rmatmul__(self, other: TensorLike) -> "Tensor":
        return _to_var(other) @ self

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __pow__(self, exponent: TensorLike) -> "Tensor":
        exponent = _to_var(exponent)
        assert type(self.data) is np.ndarray
        assert type(exponent.data) is np.ndarray
        result = Tensor(
            tensor_pow(self.data, exponent.data),
            requires_grad=(self.requires_grad or exponent.requires_grad),
            _op="pow",
        )
        if self.requires_grad:

            def _calc_grad_pow_self(x):
                return grad_pow(x, self.data, exponent.data, is_a=True)

            result.parents.add((self, _calc_grad_pow_self))
        if exponent.requires_grad:

            def _calc_grad_pow_other(x):
                return grad_pow(x, self.data, exponent.data, is_a=False)

            result.parents.add((exponent, _calc_grad_pow_other))
        result._nodes_edges.add(self)
        result._nodes_edges.add(exponent)
        # print(f"{result.shape} = {result._op}({self.label}, {exponent.label})")
        return result

    @property
    def T(self) -> "Tensor":
        result = Tensor(
            tensor_transpose(self.data), requires_grad=self.requires_grad, _op="T"
        )
        if self.requires_grad:

            def _calc_grad_transpose(x):
                return grad_transpose(x)

            result.parents.add((self, _calc_grad_transpose))
        result._nodes_edges.add(self)
        # print(f"{result.shape} = {result._op}({self.shape}")
        return result

    def sqrt(self) -> "Tensor":
        result = Tensor(
            tensor_sqrt(self.data), requires_grad=self.requires_grad, _op="sqrt"
        )
        if self.requires_grad:

            def _calc_grad_sqrt(x):
                return grad_sqrt(x, self.data)

            result.parents.add((self, _calc_grad_sqrt))
        result._nodes_edges.add(self)
        # print(f"{result.shape} = {result._op}({self.shape}")
        return result

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        result_value = tensor_sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_value, requires_grad=self.requires_grad, _op="sum")

        if self.requires_grad:

            def _grad_sum(_data: np.ndarray) -> np.ndarray:
                return grad_sum(_data, self.data, axis, keepdims)

            result.parents.add((self, _grad_sum))

        result._nodes_edges.add(self)
        # print(f"{result.shape} = {result._op}({self.shape})")
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


def trace_forward(root: Tensor):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._nodes_edges:
                edges.add((child, v))
                build(child)  # recursive

    build(root)
    return nodes, edges


def get_label(node):
    if not node.label:
        if node.requires_grad:
            return "x"
        else:
            if node.dim == 0:
                return str(node.data)
            else:
                return str(node.shape)
    else:
        return node.label


def get_formula(root: Tensor):
    visited = set()
    op_list = []

    def build(v: Tensor):
        if v not in visited:
            visited.add(v)

            if v._op:
                inputs_str = ", ".join(
                    f"{get_label(child)}" for child in v._nodes_edges
                )
                shapes_str = " x ".join(f"{child.shape}" for child in v._nodes_edges)
                op_list.append(
                    (f"{get_label(v)} = {v._op}({inputs_str}) \t\t {shapes_str}")
                )

            for child in v._nodes_edges:
                build(child)

    build(root)

    return op_list


def get_graph(root: Tensor):
    dot = graphviz.Digraph(graph_attr={"rankdir": "LR"})

    nodes, edges = trace_forward(root)
    for n in nodes:
        uid = str(id(n))
        if n.requires_grad:
            has_grad = "grad enabled"
            n_name = n.__class__.__name__
        else:
            has_grad = "constant"

            n_name = n.__class__.__name__
            if n.shape == ():
                n_name = str(n.data)
        dot.node(
            name=uid,
            label=f"{{ {n_name} | {{ shape {n.shape} | {has_grad} }} }}",
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


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
