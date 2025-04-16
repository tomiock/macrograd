from __future__ import annotations

from typing import Optional, Hashable, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto

import numpy as np


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
    SQRT = auto()
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
    succesors: set[Hashable] = field(default_factory=set)
    data: Any = None
    device: str = "cpu"
    shape: Optional[tuple[int, ...]] = None
    dtype: Optional[Type] = None
    static_data = None

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
            f"data={self.data!r}), "
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


def get_list_shape(data: Any) -> tuple[int, ...]:
    if not isinstance(data, list):
        if not isinstance(data, (int, float)):
            raise TypeError(f"Invalid type given: {type(data)}")
        return ()

    if not data:
        return (0,)

    try:
        first_element_shape = get_list_shape(data[0])
    except (ValueError, TypeError) as e:
        raise type(e)(f"Error processing element at index 0: {e}") from e

    for i, element in enumerate(data[1:], start=1):
        try:
            element_shape = get_list_shape(element)
            if element_shape != first_element_shape:
                raise ValueError(
                    f"Inconsistent shape: Element at index 0 implies sub-shape {first_element_shape}, "
                    f"but element at index {i} has sub-shape {element_shape}."
                )
        except (ValueError, TypeError) as e:
            raise type(e)(f"Error processing element at index {i}: {e}") from e
    return (len(data),) + first_element_shape


def calc_broadcast_shape(
    shape1: tuple[int, ...], shape2: tuple[int, ...]
) -> tuple[int, ...] | None:
    if shape1 == shape2:
        return shape1

    len1 = len(shape1)
    len2 = len(shape2)
    max_len = max(len1, len2)

    result_shape_reversed = []

    for i in range(1, max_len + 1):
        dim1 = shape1[-i] if i <= len1 else 1
        dim2 = shape2[-i] if i <= len2 else 1

        if dim1 == dim2:
            result_shape_reversed.append(dim1)
        elif dim1 == 1:
            result_shape_reversed.append(dim2)
        elif dim2 == 1:
            result_shape_reversed.append(dim1)
        else:
            return None  # error

    return tuple(result_shape_reversed[::-1])


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
    ) -> Hashable:
        new_id = self._get_next_id(op.name)

        # TODO: shape inference

        inf_shape = None
        inf_dtype = None
        if op == Ops.CONST:
            if isinstance(static_data, np.ndarray):
                inf_shape = static_data.shape
                inf_dtype = dtype2tensor_type(static_data.dtype)
            elif isinstance(static_data, (list, int, float)):
                inf_shape = get_list_shape(static_data)
                inf_dtype = Type.FLOAT32
            else:
                raise ValueError(f"unsuported type for data given: {type(static_data)}")

        elif op == Ops.ADD or Ops.MUL:
            if len(input_ids) == 1:
                input_ids = (input_ids[0], input_ids[0])
            elif len(input_ids) == 2:
                pass
            else:
                raise ValueError(f"Unsuported number of inputs to {op}")
            shape0 = self.nodes[input_ids[0]].shape
            shape1 = self.nodes[input_ids[1]].shape

            inf_shape = calc_broadcast_shape(shape0, shape1)
            if inf_shape is None:
                raise ValueError(f"Incompatible shapes for {op}: {shape0} and {shape1}")

        # end shape inference

        node = Node(
            id=new_id, op=op, inputs=input_ids, shape=inf_shape, dtype=inf_dtype
        )
        self.nodes[new_id] = node

        for input_id in input_ids:
            if input_id in self.nodes:
                self.nodes[input_id].succesors.add(new_id)

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
            if node.op == Ops.CONST and node.static_data is not None:
                static_data_repr = repr(node.static_data)
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
        if not hasattr(node, "succesors") or not isinstance(node.succesors, set):
            raise AttributeError("Node is missing 'succesors' method")

        for successor_id in node.succesors:
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

        for v_id in node_u.succesors:
            in_degree[v_id] -= 1

            if in_degree[v_id] == 0:
                queue.append(v_id)

    if len(result) != len(graph):
        raise ValueError("the graph appears to have cycles, this is not supported")

    return result
