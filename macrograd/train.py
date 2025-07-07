from typing import Any, Hashable

from macrograd.engine import get_default_graph
from macrograd import Tensor


class train:
    def __init__(self, func_forward):
        self.func_forward = func_forward

        self._is_allocated = False

        # WARNING: the default graph is being used
        self.graph = get_default_graph()

        self._input_nodes_ids: list[Hashable]
        self._output_node_id: Hashable

    def __call__(self, *args):
        if self._is_allocated:
            return self.allocated_call(*args)
        else:
            return self.init_call(*args)

    def init_call(self, *args):
        input_tensors = []
        for data_item in args:
            if not isinstance(data_item, Tensor):
                data_item = Tensor(data_item, node_type="data")
            input_tensors.append(data_item)

        self._input_nodes_ids = [t.node_id for t in input_tensors]

        output_tensor = self.func_forward(*input_tensors)

        self._output_node_id = output_tensor.node_id
        self.graph.allocate(backend="numpy")
        self.graph.realize()

        self._is_allocated = True
        return output_tensor

    def allocated_call(self, *args):
        if not self._is_allocated:
            raise RuntimeError

        assert len(args) == len(self._input_nodes_ids)

        for i, node_id in enumerate(self._input_nodes_ids):
            data_item = args[i]
            self.graph.nodes[node_id].computed_tensor = data_item

        self.graph.realize()

        return Tensor(_node_id=self._output_node_id)
