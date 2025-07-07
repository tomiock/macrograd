from typing import Hashable, Optional, Any, Iterable
from warnings import warn
from macrograd.engine import Graph, NodeType, get_default_graph
from macrograd.tensor import Tensor

import numpy as np


class Layer:
    def __init__(self, graph: Optional[Graph]):
        if graph is None:
            graph = get_default_graph()
        self.graph = graph

        self._params: list[Tensor] = []

    def parameters(self):
        return self._params

    def _setter_graph(self, graph: Graph):
        self.graph = graph


class Linear(Layer):
    def __init__(
        self,
        in_dims,
        out_dims,
        graph: Optional[Graph] = None,
        initilization: Optional[str] = None,
    ):
        super().__init__(graph=graph)

        self.in_dims = in_dims
        self.out_dims = out_dims
        self._w_shape = (in_dims, out_dims)
        self._b_shape = (1, out_dims)

        if initilization:
            warn("Different types of initilization not supported yet")

        w_val = np.random.randn(in_dims, out_dims).astype(np.float32) * 0.01
        b_val = np.zeros((out_dims,), dtype=np.float32)

        self.weight = Tensor(w_val, node_type="param", requires_grad=True)
        self.bias = Tensor(b_val, node_type="param", requires_grad=True)
        self._params.extend([self.weight, self.bias])

    def __call__(self, data: Tensor) -> Tensor:
        return (data @ self.weight) + self.bias


class Model:
    def __init__(self, graph: Optional[Graph] = None):
        if graph is None:
            self.graph = get_default_graph()
        else:
            self.graph = graph
        self._is_allocated = False
        self._input_nodes_ids: list[Hashable]
        self._output_node_id: Hashable
        self._parameters_nodes_ids: list[Hashable] = []

    def __call__(self, *args, **kwargs):
        if not self._is_allocated:
            return self._build_and_execute_graph(*args, **kwargs)
        else:
            return self._execute_graph(*args, **kwargs)

    def _build_and_execute_graph(self, input_data: Iterable):
        input_tensors = []

        for data_item in input_data:
            if isinstance(data_item, Tensor):
                # here we are using already created data tensors
                input_tensors.append(data_item)
            else:
                t = Tensor(data_item, node_type="data")
                input_tensors.append(t)

        self._input_nodes_ids = [t.node_id for t in input_tensors]

        # call the forward method defined by the user
        output_tensor = self.forward(*input_data)
        self._output_node_id = output_tensor.node_id

        self.graph.allocate(backend="numpy")
        self._is_allocated = True
        self.graph.realize()

        return output_tensor

    def forward(self) -> Tensor:
        raise NotImplementedError("This should be defined by the user")

    def _execute_graph(self, data: Any):
        if not self._is_allocated:
            raise RuntimeError

        input_data_dict = {}
        for i, node_id in enumerate(self._input_nodes_ids):
            print(node_id)
            data_item = data[i]

            if isinstance(data_item, Tensor):
                input_data_dict[node_id] = data_item.data
            else:
                input_data_dict[node_id] = data_item

        self.graph.realize()

        return Tensor(_node_id=self._output_node_id)

    def _collect_parameters_nodes(self):
        for node_id, node in self.graph.nodes.items():
            if node.type == NodeType.PARAM:
                self._parameters_nodes_ids.append(node_id)

    def zero_params(self):
        for node in self._parameters_nodes_ids:
            self.graph.nodes[node].grad = np.zeros_like(self.graph.nodes[node].grad)

    def _setter_layers_graph(self):
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        if not self._parameters_nodes_ids:
            self._collect_parameters_nodes()

        return [Tensor(_node_id=id) for id in self._parameters_nodes_ids]
