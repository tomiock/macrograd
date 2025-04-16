from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Hashable, Optional

import numpy as np

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
    POW = auto()
    CONV2D = auto()
    CONV1D = auto()

    # views
    TRANSPOSE = auto()
    RESHAPE = auto()

    # for constants (data given)
    CONST = auto()


class Type(FastEnum):
    INT64 = auto()
    INT32 = auto()
    FLOAT64 = auto()
    FLOAT32 = auto()


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
                inf_dtype = Type.FLOAT32  # TODO: not hardcoded
            else:
                raise ValueError(f"Unsuported type for data given: {type(static_data)}")

        elif op == Ops.ADD or op == Ops.MUL:
            shape0 = self.nodes[input_ids[0]].shape
            shape1 = self.nodes[input_ids[1]].shape
            if shape1 is None or shape0 is None:
                raise ValueError("Could not found shape of previous tensor")

            try:
                inf_shape = _calc_broadcast_shape(shape0, shape1)
            except ValueError:
                raise ValueError(f"Incompatible shapes for {op}: {shape0} and {shape1}")

        elif op == Ops.MATMUL:
            shape0 = self.nodes[input_ids[0]].shape
            shape1 = self.nodes[input_ids[1]].shape
            if shape1 is None or shape0 is None:
                raise ValueError("Could not found shape of previous tensor")

            try:
                # TODO: support batched operations
                inf_shape = _calc_matmul_shape(shape0, shape1)
            except ValueError:
                raise ValueError(f"Incompatible shapes for {op}: {shape0} and {shape1}")

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

        elif op == Ops.SUM:
            shape_in = self.nodes[input_ids[0]].shape
            if shape_in is None:
                raise ValueError(f"Shape for node {input_ids[0]} is None")

            try:
                axis = kwargs["axis"]
                keepdims = kwargs["keepdims"]
                inf_shape = _calc_reduction_shape(shape_in, axis, keepdims)
            except KeyError:
                pass

        # end shape inference

        node = Node(
            id=new_id,
            op=op,
            inputs=input_ids,
            shape=inf_shape,
            dtype=inf_dtype,
            op_kwargs=kwargs,
        )
        self.nodes[new_id] = node

        for input_id in input_ids:
            if input_id in self.nodes:
                self.nodes[input_id].successors.add(new_id)

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
            label = f"ID: {node.id!r}\nOp: {op_name}\nShape: {node.shape}\nDtype: {node.dtype}"
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


def topo_sort(graph: dict[Hashable, Node]) -> list[Hashable]:
    in_degree: dict[Hashable, int] = defaultdict(int)

    for _, node in graph.items():
        if not hasattr(node, "succesors") or not isinstance(node.successors, set):
            raise AttributeError("Node is missing 'succesors' method")

        for successor_id in node.successors:
            if successor_id not in graph:
                raise KeyError("node in succesors not found in graph")
            in_degree[successor_id] += 1

    # initial queue with all nodes that do not have succesors (source nodes)
    queue = deque([node_id for node_id in graph if in_degree[node_id] == 0])

    result: list[Hashable] = []

    while queue:
        u_id = queue.popleft()
        result.append(u_id)

        node_u = graph[u_id]

        for v_id in node_u.successors:
            in_degree[v_id] -= 1

            if in_degree[v_id] == 0:
                queue.append(v_id)

    if len(result) != len(graph):
        raise ValueError("the graph appears to have cycles, this is not supported")

    return result
