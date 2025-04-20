from __future__ import annotations

from math import prod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Hashable, Optional

import numpy as np
import cupy as cp

from macrograd.utils_shape import (
    _calc_broadcast_shape,
    _calc_matmul_shape,
    _calc_reduction_shape,
    _get_list_shape,
)


class FastEnum(IntEnum):
    def __str__(self):
        return Enum.__str__(self)

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])


class Ops(FastEnum):
    # numerical
    ADD = auto()
    MUL = auto()
    MATMUL = auto()
    SUM = auto()
    MAX = auto()
    EXP = auto()
    LOG = auto()
    POW = auto()
    CONV2D = auto()
    CONV1D = auto()

    # views
    TRANSPOSE = auto()
    RESHAPE = auto()

    # for constants (data given)
    CONST = auto()


class Type(FastEnum):
    INT32 = 10
    INT64 = 20
    FLOAT32 = 30
    FLOAT64 = 40


def calc_promoted_dtype(dtype1: Type, dtype2: Type) -> Type:
    promoted_rank = max(dtype2.value, dtype1.value)

    try:
        return Type(promoted_rank)
    except ValueError:
        raise ValueError("Internal Error, did not found corresponding type")


def unary_dtype_handle(node_in):
    if node_in.dtype in (Type.INT32, Type.INT64):
        inf_dtype = Type.FLOAT32
    elif node_in.dtype in (Type.FLOAT32, Type.FLOAT64):
        inf_dtype = node_in.dtype
    else:
        raise TypeError

    return inf_dtype


def dtype2tensor_type(dtype: np.dtype):
    if dtype == np.float32:
        return Type.FLOAT32
    if dtype == np.float64:
        return Type.FLOAT64
    elif dtype == np.int64:
        return Type.INT64
    elif dtype == np.int32:
        return Type.INT32
    else:
        raise TypeError("Invalid type taken from numpy")


@dataclass
class Node:
    id: Hashable
    op: Optional[Ops] = None
    inputs: tuple[Hashable, ...] = field(default_factory=tuple)
    successors: set[Hashable] = field(default_factory=set)
    shape: Optional[tuple[int, ...]] = None
    dtype: Optional[Type] = None
    device: str = "cpu"
    computed_tensor: Optional[np.ndarray] = None

    op_kwargs: dict[str, Any] = field(default_factory=dict)

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
        successor_ids = ", ".join(map(repr, self.successors))

        return (
            f"{self.__class__.__name__}("
            f"id={self.id!r}, op={op_name}, "
            f"inputs=({input_ids}), "
            f"successors={{{successor_ids}}}, "
            f"data={self.computed_tensor!r}), "
            f"shape={self.shape!r})"
        )


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


class Graph:
    def __init__(self):
        # the order here is the order of creation
        self.nodes: dict[Hashable, Node] = {}
        self._op_counters: dict[str, int] = defaultdict(int)

    def _get_next_id(self, op_name: str) -> str:
        count = self._op_counters[op_name]
        self._op_counters[op_name] += 1
        new_id = f"{op_name.lower()}_{count}"

        # TODO: do collision check

        return new_id

    def realize(self) -> None:
        exec

    def add_node(
        self,
        op: Ops,
        input_ids: tuple[Hashable, ...],
        # supported: float, int, list, ndarray
        static_data: Any = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Hashable:
        new_id = self._get_next_id(op.name)

        # TODO: shape inference

        inf_shape = None
        inf_dtype = None

        if kwargs is None:
            kwargs = defaultdict()

        if op == Ops.CONST:
            if isinstance(static_data, np.ndarray):
                inf_shape = static_data.shape
                inf_dtype = dtype2tensor_type(static_data.dtype)
            elif isinstance(static_data, (list, int, float)):
                inf_shape = _get_list_shape(static_data)
                if isinstance(static_data, float):
                    inf_dtype = Type.FLOAT32
                elif isinstance(static_data, int):
                    inf_dtype = Type.INT32
                elif isinstance(static_data, list):
                    inf_dtype = Type.FLOAT32
            else:
                raise TypeError(f"Unsuported type for data given: {type(static_data)}")

        elif op == Ops.ADD or op == Ops.MUL:
            node0 = self.nodes[input_ids[0]]
            node1 = self.nodes[input_ids[1]]

            shape0 = node0.shape
            shape1 = node1.shape
            type0 = node0.dtype
            type1 = node1.dtype

            if shape1 is None or shape0 is None or type0 is None or type1 is None:
                raise ValueError(
                    f"Could not found metadata for the input tensors for nodes {input_ids}"
                )

            try:
                inf_shape = _calc_broadcast_shape(shape0, shape1)
            except ValueError:
                raise ValueError(f"Incompatible shapes for {op}: {shape0} and {shape1}")

            inf_dtype = calc_promoted_dtype(type0, type1)

        elif op == Ops.MATMUL:
            node0 = self.nodes[input_ids[0]]
            node1 = self.nodes[input_ids[1]]

            shape0 = node0.shape
            shape1 = node1.shape
            type0 = node0.dtype
            type1 = node1.dtype

            if shape1 is None or shape0 is None or type0 is None or type1 is None:
                raise ValueError(
                    f"Could not found metadata for the input tensors for nodes {input_ids}"
                )

            try:
                # TODO: support batched operations
                inf_shape = _calc_matmul_shape(shape0, shape1)
            except ValueError:
                raise ValueError(f"Incompatible shapes for {op}: {shape0} and {shape1}")

            inf_dtype = calc_promoted_dtype(type0, type1)

        elif op == Ops.POW:
            shape_exp = self.nodes[input_ids[1]].shape
            shape_base = self.nodes[input_ids[0]].shape

            # applied element wise into the base tensor
            inf_shape = shape_base
            if shape_exp is None:
                raise ValueError("Exponent has no shape")

            for dim_size in shape_exp:
                if dim_size != 1:
                    raise ValueError(
                        f"Exponent tensor has to be scalar-like: {shape_base} shape found"
                    )

            type1 = self.nodes[input_ids[1]].dtype
            type0 = self.nodes[input_ids[0]].dtype
            if type1 is None or type0 is None:
                raise TypeError(f"Unsuported types in {op}: {type1}, {type0}")
            try:
                inf_dtype = calc_promoted_dtype(type0, type1)
            except TypeError as e:
                raise e

        elif op == Ops.SUM or op == Ops.MAX:
            node_in = self.nodes[input_ids[0]]

            if node_in.shape is None:
                raise ValueError(f"Shape for node {input_ids[0]} is None")

            try:
                axis = kwargs["axis"]
                keepdims = kwargs["keepdims"]
                inf_shape = _calc_reduction_shape(node_in.shape, axis, keepdims)
            except KeyError:
                pass

            try:
                inf_dtype = unary_dtype_handle(node_in)
            except TypeError:
                raise TypeError(f"Unsuported type for {op}: {node_in.dtype}")

        elif op == Ops.EXP:
            node_in = self.nodes[input_ids[0]]
            shape_in = node_in.shape
            if shape_in is None:
                raise ValueError(f"Shape for node {input_ids[0]} is None")
            inf_shape = shape_in

            try:
                inf_dtype = unary_dtype_handle(node_in)
            except TypeError:
                raise TypeError(f"Unsuported type for {op}: {node_in.dtype}")

        elif op == Ops.LOG:
            node_in = self.nodes[input_ids[0]]
            shape_in = node_in.shape

            if kwargs["base"] is None:
                raise ValueError("Was not provided `base` to log operation")

            if shape_in is None:
                raise ValueError(f"Shape for node {input_ids[0]} is None")
            inf_shape = shape_in

            try:
                inf_dtype = unary_dtype_handle(node_in)
            except TypeError:
                raise TypeError(f"Unsuported type for {op}: {node_in.dtype}")

        elif op == Ops.RESHAPE:
            node_in = self.nodes[input_ids[0]]
            shape_in = node_in.shape
            inf_dtype = node_in.dtype

            if shape_in is None:
                raise ValueError("Shape of input tensor is none")

            if kwargs.get("shape"):
                new_shape_arg = kwargs.get("shape")
            else:
                raise ValueError("Required `shape` argument in reshape operation")

            number_elements = prod(shape_in)

            if isinstance(new_shape_arg, int):
                if (new_shape_arg == number_elements) or (new_shape_arg == -1):
                    inf_shape = (number_elements,)
                else:
                    raise ValueError(
                        f"Missmatch in shape, tensor with {number_elements} elements cannot be reshape into {(new_shape_arg,)}"
                    )
            elif isinstance(new_shape_arg, tuple):
                product_known_dims = 1
                unknown_dim_count = 0
                unknown_dim_index = -1

                for i, dim in enumerate(new_shape_arg):
                    if not isinstance(dim, int):
                        raise TypeError(f"Target dim must be `int`, got {type(dim)}")

                    if dim == -1:
                        unknown_dim_count += 1
                        unknown_dim_index = i
                    elif dim < 0:
                        raise ValueError(
                            f"Target dim cannot be negative, got {dim} from {new_shape_arg}"
                        )
                    else:
                        if dim == 0 and number_elements != 0:
                            raise ValueError(
                                f"Target shape cannot have a zero, got {new_shape_arg}"
                            )
                        if dim > 0:
                            product_known_dims *= dim
                if unknown_dim_count == 0:
                    if product_known_dims != number_elements:
                        if new_shape_arg == () and number_elements == 1:
                            inf_shape = ()
                        else:
                            raise ValueError(
                                f"Cannot reshape {shape_in} to {new_shape_arg}"
                            )

                    else:
                        inf_shape = new_shape_arg
                elif unknown_dim_count == 1:
                    if product_known_dims == 0:
                        if number_elements != 0:
                            raise ValueError(
                                f"Cannot reshape {shape_in} to {new_shape_arg}"
                            )
                        inf_dim = 0
                    else:
                        if number_elements % product_known_dims != 0:
                            raise ValueError(
                                f"Cannot reshape {shape_in} to {new_shape_arg}"
                            )
                        inf_dim = number_elements // product_known_dims

                    inf_shape_list = list(new_shape_arg)
                    inf_shape_list[unknown_dim_index] = inf_dim
                    inf_shape = tuple(inf_shape_list)
                else:
                    raise ValueError(
                        "Only one (-1) in the new_shape pls, got {new_shape_arg}"
                    )
        elif op == Ops.TRANSPOSE:
            node_in = self.nodes[input_ids[0]]
            shape_in = node_in.shape
            inf_dtype = node_in.dtype

            if not shape_in:
                raise ValueError("Shape of input is None")

            ndim = len(shape_in)

            axes = kwargs.get("axes", None)

            if axes is None:
                axes = tuple(range(ndim))[::-1]
            else:
                if not isinstance(axes, (tuple, list)):
                    raise TypeError(
                        f"Tranpose axes given must be `tuple` or `list`, got: {type(axes)}"
                    )
                axes = tuple(axes)

                if len(axes) != ndim:
                    raise ValueError(
                        f"Tranpose axes len mush much the ndim of tensor, got {len(axes)}, expected {ndim}"
                    )

                seen_axes = set()
                for ax in axes:
                    if not isinstance(ax, int):
                        raise TypeError(
                            f"Tranpose axes must have only ints, got {ax} of type {type(ax)}"
                        )
                    if not (0 <= ax < ndim):
                        raise ValueError(f"Axes out of bounds {ax}")
                    if ax in seen_axes:
                        raise ValueError(f"Duplicate axes: {ax}")

                    seen_axes.add(ax)

                if len(seen_axes) != ndim:
                    raise ValueError("Tranpose axes missmatch")

            inf_shape = tuple(shape_in[i] for i in axes)

        else:
            raise ValueError("Unsuported OP found: {op}, internal error")

        node = Node(
            id=new_id,
            op=op,
            inputs=input_ids,
            shape=inf_shape,
            dtype=inf_dtype,
            op_kwargs=kwargs,
        )

        for input_id in input_ids:
            if input_id in self.nodes:
                self.nodes[input_id].successors.add(new_id)

        self.nodes[new_id] = node
        node.computed_tensor = static_data

        return new_id

    def __repr__(self) -> str:
        return " \n".join(map(repr, self.nodes))

    def visualize(
        self,
        filename: str = "computation_graph.gv",
        view: bool = True,
        format: str = "png",
    ):
        """
        Generates a visualization of the graph using Graphviz.

        Args:
            filename (str): The base name for the output file (without extension).
            view (bool): If True, try to open the rendered graph automatically.
            format (str): The output format (e.g., 'png', 'svg', 'pdf').

        Requires the 'graphviz' Python package and the Graphviz system library.
        """
        try:
            import graphviz
        except ImportError:
            print(
                "Error: 'graphviz' package not found. Please install it (`pip install graphviz`)"
            )
            print("       and ensure the Graphviz system library is installed.")
            return None  # Or raise ImportError

        dot = graphviz.Digraph(comment="Computation Graph")
        dot.attr(rankdir="TB")  # Top-to-Bottom layout

        added_nodes = set()

        for node_id, node in self.nodes.items():
            node_id_str = str(node_id)  # Ensure string for graphviz
            added_nodes.add(node_id_str)

            op_name = node.op.name if node.op else "const"
            # Create a multi-line label for the node

            if node.op_kwargs:
                label = f"ID: {node.id!r}\nop: {op_name}\nshape: {node.shape}\ndtype: {node.dtype}\nkwargs:\n{node.op_kwargs}"
            else:
                label = f"ID: {node.id!r}\nop: {op_name}\nshape: {node.shape}\ndtype: {node.dtype}"

            if node.op == Ops.CONST and node.computed_tensor is not None:
                static_data_repr = repr(node.computed_tensor)
                if len(static_data_repr) > 30:
                    static_data_repr = static_data_repr[:27] + "..."
                label += f"\nValue: {static_data_repr}"

            # Use different shapes for source vs op nodes for clarity (optional)
            shape_style = "ellipse" if node.op else "box"

            dot.node(node_id_str, label=label, shape=shape_style)

        for node_id, node in self.nodes.items():
            child_id_str = str(node_id)
            for input_id in node.inputs:
                parent_id_str = str(input_id)
                # Ensure parent node exists before adding edge
                if parent_id_str in added_nodes:
                    dot.edge(parent_id_str, child_id_str)
                else:
                    print(
                        f"Warning: Input node '{parent_id_str}' for node '{child_id_str}' not found in graph for visualization."
                    )

        try:
            rendered_path = dot.render(filename, view=view, format=format, cleanup=True)
            print(f"Graph saved to {rendered_path}.{format}")
            return dot
        except Exception as e:
            print(f"Error rendering graph with Graphviz: {e}")
            print(
                "Ensure the Graphviz system library (dot command) is installed and in your PATH."
            )
            return None


def topo_sort(graph_nodes: dict[Hashable, Node]) -> list[Hashable]:
    in_degree: dict[Hashable, int] = defaultdict(int)

    for _, node in graph_nodes.items():
        if not hasattr(node, "successors") or not isinstance(node.successors, set):
            raise AttributeError("Node is missing 'succesors' method")

        for successor_id in node.successors:
            if successor_id not in graph_nodes:
                raise KeyError("node in succesors not found in graph")
            in_degree[successor_id] += 1

    # initial queue with all nodes that do not have succesors (source nodes)
    queue = deque([node_id for node_id in graph_nodes if in_degree[node_id] == 0])

    result: list[Hashable] = []

    while queue:
        u_id = queue.popleft()
        result.append(u_id)

        node_u = graph_nodes[u_id]

        for v_id in node_u.successors:
            in_degree[v_id] -= 1

            if in_degree[v_id] == 0:
                queue.append(v_id)

    if len(result) != len(graph_nodes):
        raise ValueError(
            "the graph appears to have cycles, this is not supported. wtf did you do, no RNNs allowed here."
        )

    return result


def executor(graph: Graph) -> np.ndarray:
    exec_list = topo_sort(graph.nodes)

    for node_id in exec_list:
        node = graph.nodes[node_id]

        if node.op == Ops.CONST:
            pass

        elif node.op == Ops.ADD:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            tensor1 = graph.nodes[node.inputs[1]].computed_tensor

            node.computed_tensor = np.add(tensor0, tensor1)

        elif node.op == Ops.MUL:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            tensor1 = graph.nodes[node.inputs[1]].computed_tensor

            node.computed_tensor = np.multiply(tensor0, tensor1)

        elif node.op == Ops.EXP:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor

            node.computed_tensor = np.exp(tensor0)

        elif node.op == Ops.SUM:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor

            keepdims = node.op_kwargs.get("keepdims")
            axis = node.op_kwargs.get("axis")

            node.computed_tensor = np.sum(tensor0, axis=axis, keepdims=keepdims)

        elif node.op == Ops.POW:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            tensor1 = graph.nodes[node.inputs[1]].computed_tensor

            node.computed_tensor = np.pow(tensor0, tensor1)

        elif node.op == Ops.LOG:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            base = node.op_kwargs.get("base")

            if base == "e":
                node.computed_tensor = np.log(tensor0)
            else:
                node.computed_tensor = np.log(tensor0) / np.log(base)

        elif node.op == Ops.MATMUL:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            tensor1 = graph.nodes[node.inputs[1]].computed_tensor

            node.computed_tensor = np.matmul(tensor0, tensor1)

        elif node.op == Ops.RESHAPE:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            shape = node.op_kwargs.get("shape")

            node.computed_tensor = np.reshape(tensor0, shape=shape)

        elif node.op == Ops.TRANSPOSE:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            axes = node.op_kwargs.get("axes")

            node.computed_tensor = np.transpose(tensor0, axes=axes)

        else:
            raise RuntimeError(f"Op {node.op} not supported")


def executor_cupy(graph: Graph) -> cp.ndarray:
    exec_list = topo_sort(graph.nodes)

    for node_id in exec_list:
        node = graph.nodes[node_id]

        if node.op == Ops.CONST:
            node.computed_tensor = cp.array(node.computed_tensor)

        elif node.op == Ops.ADD:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            tensor1 = graph.nodes[node.inputs[1]].computed_tensor

            node.computed_tensor = cp.add(tensor0, tensor1)

        elif node.op == Ops.MUL:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            tensor1 = graph.nodes[node.inputs[1]].computed_tensor

            node.computed_tensor = cp.multiply(tensor0, tensor1)

        elif node.op == Ops.EXP:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor

            node.computed_tensor = cp.exp(tensor0)

        elif node.op == Ops.SUM:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor

            keepdims = node.op_kwargs.get("keepdims")
            axis = node.op_kwargs.get("axis")

            node.computed_tensor = cp.sum(tensor0, axis=axis, keepdims=keepdims)

        elif node.op == Ops.POW:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            tensor1 = graph.nodes[node.inputs[1]].computed_tensor

            node.computed_tensor = cp.pow(tensor0, tensor1)

        elif node.op == Ops.LOG:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            base = node.op_kwargs.get("base")

            if base == "e":
                node.computed_tensor = cp.log(tensor0)
            else:
                node.computed_tensor = cp.log(tensor0) / cp.log(base)

        elif node.op == Ops.MATMUL:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            tensor1 = graph.nodes[node.inputs[1]].computed_tensor

            node.computed_tensor = cp.matmul(tensor0, tensor1)

        elif node.op == Ops.RESHAPE:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            shape = node.op_kwargs.get("shape")

            node.computed_tensor = cp.reshape(tensor0, shape=shape)

        elif node.op == Ops.TRANSPOSE:
            tensor0 = graph.nodes[node.inputs[0]].computed_tensor
            axes = node.op_kwargs.get("axes")

            node.computed_tensor = cp.transpose(tensor0, axes=axes)

        else:
            raise RuntimeError(f"Op {node.op} not supported")
