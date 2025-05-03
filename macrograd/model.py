from typing import Hashable, Optional, Any, Iterable
from warnings import warn
import numpy as np
from macrograd.engine import Graph, NodeType, get_default_graph
from macrograd.tensor import Tensor


class Layer:
    def __init__(self, graph: Optional[Graph]):
        if graph is None:
            graph = get_default_graph()
        self.graph = graph

        self._params: list[Tensor] = []

    @property
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

    def __call__(self, *args):
        if not self._is_allocated:
            return self._build_and_execute_graph(args)
        else:
            return self._execute_graph(args)

    def _build_and_execute_graph(self, input_data: Iterable):
        input_tensors = []
        for i, data_item in enumerate(input_data):
            if isinstance(data_item, Tensor):
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

    def forward(self, *args) -> Tensor:
        raise NotImplementedError("This should be defined by the user")

    def _execute_graph(self, data: Any):
        if not self._is_allocated:
            raise RuntimeError

        input_data_dict = {}
        for i, node_id in enumerate(self._input_nodes_ids):
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

    def _setter_layers_graph(self):
        raise NotImplementedError

    @property
    def parameters(self) -> list[Tensor]:
        if not self._parameters_nodes_ids:
            self._collect_parameters_nodes()

        return [Tensor(_node_id=id) for id in self._parameters_nodes_ids]


class SGD_Optimizer:
    def __init__(
        self, learning_rate: float = 0.01, minimizing=True, step_function=None
    ):
        self.lr = learning_rate
        self.min = minimizing
        self.step_fn = step_function

        if not self.step_fn:
            if self.min:

                def step_fn(param):
                    param.data -= param.grad * self.lr
                    return param
            else:

                def step_fn(param):
                    param.data += param.grad * self.lr
                    return param

            self.step_fn = step_fn

    def step(self, loss: Tensor, params: list[Tensor]) -> list[Tensor]:
        loss.backprop()

        return list(map(self.step_fn, params))


class SGD_MomentumOptimizer:
    def __init__(
        self, learning_rate: float = 0.01, alpha: float = 0.9, params_copy: list = []
    ):
        self.lr = learning_rate
        self.velocities = []

        if 0 <= alpha < 1:
            self.alpha = Tensor(alpha, requires_grad=False)
        else:
            raise ValueError("Alpha hyperparemter must be between 0 and 1")

        for param in params_copy:
            self.velocities.append(
                Tensor(
                    np.zeros_like(param.data),
                    requires_grad=False,
                )
            )

        del params_copy

    def step_fn(self, param: Tensor, index: int):
        self.velocities[index].data = (
            self.alpha.data * self.velocities[index].data - self.lr * param.grad
        )
        param.data = param.data + self.velocities[index].data
        return param

    def step(self, loss: Tensor, params: list[Tensor]) -> list[Tensor]:
        loss.backprop()

        updated_params = []
        for index, param in enumerate(params):
            updated_params.append(self.step_fn(param, index))
        return updated_params
