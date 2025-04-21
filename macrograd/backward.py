import math
from warnings import warn
import numpy as np

from typing import Hashable, Sequence, Optional
from .engine import Graph, Ops, topo_sort, executor
from .utils_shape import _normalize_axis


def _accumulate_grad(graph: Graph, node_id: Hashable, grad_contribution: np.ndarray):
    if node_id not in graph.nodes:
        warn(f"Cannot store gradient for node {node_id} that is not on the graph")
        return

    node = graph.nodes[node_id]

    if not getattr(node, "requires_grad", False):
        return

    if node.grad is None:
        if node.shape is None:
            warn(f"Cannot init gradient for {node_id}. It does not have a shape")
            return
        if node.shape != grad_contribution.shape:
            warn("Shape missmatch between grad and node")
            print(f"Got {node.shape = } and {grad_contribution.shape}")
            return

        node.grad = np.zeros_like(grad_contribution, dtype=grad_contribution.dtype)
        node.grad += grad_contribution
    else:
        if node.grad.shape != grad_contribution.shape:
            warn(
                f"Backward Pass Warning: Shape mismatch during gradient accumulation for {node_id!r}:"
                f" existing {node.grad.shape}, new {grad_contribution.shape}. Skipping accumulation."
            )
            return
        node.grad += grad_contribution


def _backward(graph: Graph, node_id: Hashable):
    forward_exec = topo_sort(graph.nodes)

    backward_exec = reversed(forward_exec)

    nodes_to_process: list[Hashable] = []
    start_node_found = False
    for current_node_id in reversed(forward_exec):
        nodes_to_process.append(current_node_id)
        if current_node_id == node_id:
            start_node_found = True
            break
    if not start_node_found:
        raise ValueError(f"Node {node_id!r} not found in forward exec order.")

    loss_node = graph.nodes[node_id]
    np_dtype = np.float32  # Placeholder - TODO: map from loss_node.dtype
    if loss_node.shape == ():
        loss_node.grad = np.array(1.0, dtype=np_dtype)
    else:
        loss_node.grad = np.ones_like(loss_node.computed_tensor, dtype=np_dtype)

    for node_id_exec in backward_exec:
        node = graph.nodes[node_id_exec]

        if node.grad is None:
            continue

        if node.op is Ops.CONST:
            continue

        input_nodes = [graph.nodes[in_id] for in_id in node.inputs]
        forward_input_tensors = [n.computed_tensor for n in input_nodes]

        for tensor in forward_input_tensors:
            # TODO: fix this
            if not isinstance(tensor, (np.ndarray, np.float64, np.float32)):
                raise TypeError(
                    f"Need a `np.ndarray` stored on the nodes, got {type(tensor)}"
                )

        if any(t is None for t in forward_input_tensors):
            warn(
                f"Missing forward pass tensor for inputs of {node.id!r}. Skipping gradient propagation."
            )
            continue

        if node.op == Ops.ADD:
            grad0 = grad_add(node.grad, forward_input_tensors[0])
            grad1 = grad_add(node.grad, forward_input_tensors[1])
            _accumulate_grad(graph, node.inputs[0], grad0)
            _accumulate_grad(graph, node.inputs[1], grad1)

        elif node.op == Ops.MUL:
            grad0 = grad_mul(
                node.grad, forward_input_tensors[0], forward_input_tensors[1]
            )
            grad1 = grad_mul(
                node.grad, forward_input_tensors[1], forward_input_tensors[0]
            )
            _accumulate_grad(graph, node.inputs[0], grad0)
            _accumulate_grad(graph, node.inputs[1], grad1)

        elif node.op == Ops.MATMUL:
            grad0 = grad_matmul(
                node.grad,
                forward_input_tensors[0],
                forward_input_tensors[1],
                is_left=True,
            )
            grad1 = grad_matmul(
                node.grad,
                forward_input_tensors[0],
                forward_input_tensors[1],
                is_left=False,
            )
            _accumulate_grad(graph, node.inputs[0], grad0)
            _accumulate_grad(graph, node.inputs[1], grad1)

        elif node.op == Ops.POW:
            # tensor**scalar operation
            if len(node.inputs) == 1:
                base_tensor = forward_input_tensors[0]
                exponent_scalar = node.op_kwargs.get("exponent")
                if exponent_scalar is None:
                    raise ValueError("Scalar exponent missing for POW")
                exponent_arr = np.array(exponent_scalar)

                grad_base = grad_pow(node.grad, base_tensor, exponent_arr, is_a=True)

            # tensor**tensor operation
            elif len(node.inputs) == 2:
                base_tensor = forward_input_tensors[0]
                exponent_tensor = forward_input_tensors[1]
                grad_base = grad_pow(node.grad, base_tensor, exponent_tensor, is_a=True)
                grad_exponent = grad_pow(
                    node.grad, base_tensor, exponent_tensor, is_a=False
                )
                grad_exponent = grad_exponent.reshape(exponent_tensor.shape)
            _accumulate_grad(graph, node.inputs[0], grad_base)
            _accumulate_grad(graph, node.inputs[1], grad_exponent)

        elif node.op == Ops.SUM:
            if len(node.inputs) != 1:
                continue
            in_tensor = forward_input_tensors[0]
            axis = node.op_kwargs.get("axis")
            keepdims = node.op_kwargs.get("keepdims", False)
            grad0 = grad_sum(node.grad, in_tensor, axis, keepdims)
            _accumulate_grad(graph, node.inputs[0], grad0)

        elif node.op == Ops.MAX:
            if len(node.inputs) != 1:
                continue
            in_tensor = forward_input_tensors[0]
            axis = node.op_kwargs.get("axis")
            keepdims = node.op_kwargs.get("keepdims", False)

            grad0 = grad_max(node.grad, in_tensor, axis, keepdims)
            _accumulate_grad(graph, node.inputs[0], grad0)

        elif node.op == Ops.TRANSPOSE:
            if len(node.inputs) != 1:
                continue
            axes = node.op_kwargs.get("axes", None)
            grad0 = grad_transpose(node.grad, axes)
            _accumulate_grad(graph, node.inputs[0], grad0)

        elif node.op == Ops.EXP:
            if len(node.inputs) != 1:
                continue
            if node.computed_tensor is None:
                continue
            grad0 = node.grad * node.computed_tensor
            _accumulate_grad(graph, node.inputs[0], grad0)

        elif node.op == Ops.RESHAPE:
            if len(node.inputs) != 1:
                continue
            input_node = graph.nodes[node.inputs[0]]
            if input_node.shape is None:
                continue  # Need original shape
            grad0 = node.grad.reshape(input_node.shape)
            _accumulate_grad(graph, node.inputs[0], grad0)
        
        elif node.op == Ops.RELU:
            if len(node.inputs) != 1:
                continue
            input_node = graph.nodes[node.inputs[0]]
            grad0 = grad_relu(node.grad, input_node.computed_tensor)
            _accumulate_grad(graph, node.inputs[0], grad0)

        # TODO: mode to grad_log
        elif node.op == Ops.LOG:
            if len(node.inputs) != 1:
                continue
            input_node = graph.nodes[node.inputs[0]]
            if input_node.shape is None:
                continue

            x = input_node.computed_tensor
            if x is None:
                warn("Node value stored in node")

            base = node.op_kwargs.get("base", math.e)
            if base == "e":
                grad0 = node.grad * (1.0 / x)
            else:
                ln_base = math.log(float(base))
                grad0 = node.grad * (1.0 / (x * ln_base))

            _accumulate_grad(graph, node.inputs[0], grad0)

        else:
            warn(
                f"Backward pass not implemented for Op: {node.op.name if node.op else 'SOURCE'}"
            )


def unbroadcast(grad, target_shape, broadcast_idx=0) -> np.ndarray:
    target_ndim = len(target_shape)

    while grad.ndim > target_ndim:
        grad = np.sum(grad, axis=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            grad = np.sum(grad, axis=axis, keepdims=True)
    return grad


def grad_add(_data: np.ndarray, arr1: np.ndarray) -> np.ndarray:
    return unbroadcast(_data, arr1.shape)


def grad_mul(_data: np.ndarray, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return unbroadcast(np.multiply(_data, arr2), arr1.shape)


def grad_matmul(
    _data: np.ndarray,
    arr1: np.ndarray,
    arr2: np.ndarray,
    is_left: bool,
) -> np.ndarray:
    ndim1, ndim2 = arr1.ndim, arr2.ndim
    shape1, shape2 = arr1.shape, arr2.shape

    if is_left:
        if ndim2 == 1:
            grad = _data * arr2
        else:
            axes_T = list(range(ndim2))
            # tranpose the last two axes
            if ndim2 >= 2:
                axes_T[-1], axes_T[-2] = axes_T[-2], axes_T[-1]
            arr2_T = np.transpose(arr2, axes=axes_T)
            grad = np.matmul(_data, arr2_T)
        return unbroadcast(grad, shape1)
    else:
        if ndim1 == 1:
            if _data.ndim == 0:
                grad = arr1 * _data
            else:
                x_col = arr1.reshape(-1, 1)
                dL_dy_row = _data.reshape(1, -1)
                grad = np.matmul(x_col, dL_dy_row)
        else:
            axes_T = list(range(ndim1))
            # transpose the last two axes
            if ndim1 >= 2:
                axes_T[-1], axes_T[-2] = axes_T[-2], axes_T[-1]
            arr1_T = np.transpose(arr1, axes=axes_T)
            grad = np.matmul(arr1_T, _data)
        return unbroadcast(grad, shape2)


def grad_pow(
    _data: np.ndarray,
    arr: np.ndarray,
    exponent: np.ndarray,
    is_a: bool,
) -> np.ndarray:
    if is_a:
        return unbroadcast(
            _data * exponent * arr ** np.where(exponent, exponent - 1, 1.0), arr.shape
        )
    else:
        return unbroadcast(
            _data * np.log(arr) * np.power(arr, exponent), exponent.shape
        )


def grad_sum(_data: np.ndarray, arr: np.ndarray, axis, keepdims) -> np.ndarray:
    if axis is None:
        return np.multiply(_data, np.ones_like(arr))

    if not keepdims:
        _data_expanded = np.expand_dims(_data, axis=axis)
        return np.multiply(_data_expanded, np.ones_like(arr))
    else:
        return np.multiply(_data, np.ones_like(arr))


def grad_transpose(_data: np.ndarray, axes=None) -> np.ndarray:
    return np.transpose(_data, axes=axes)


def grad_relu(_data: np.ndarray, arr: np.ndarray):
    print(_data)
    return _data * arr.data > 0


def grad_max(
    _data: np.ndarray,
    arr: np.ndarray,
    axis: Optional[int | Sequence[int]],
    keepdims: bool,
) -> np.ndarray:
    norm_axis = _normalize_axis(axis, arr.ndim)
    output_fwd = np.max(arr, axis=norm_axis, keepdims=True)
    mask = (arr == output_fwd)

    num_max = np.sum(mask, axis=norm_axis, keepdims=True)
    num_max = np.where(num_max == 0, 1.0, num_max)

    _data_expanded = _data
    if not keepdims and arr.ndim > 0 and norm_axis is not None and len(norm_axis) > 0:
        try:
            # Add back the reduced dimensions to match the broadcast shape
            _data_expanded = np.expand_dims(_data, axis=norm_axis)
        except Exception:
            raise ValueError("Shape missmatch when calculating the gradient of `max`")
    distributed_mask = _data_expanded / mask
    total_mask = mask * distributed_mask
    total_mask[np.isnan(total_mask)] = 0

    return total_mask
