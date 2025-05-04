from typing import Any, Optional, Sequence


def _get_list_shape(data: Any) -> tuple[int, ...]:
    if not isinstance(data, list):
        if not isinstance(data, (int, float)):
            raise TypeError(f"Invalid type given: {type(data)}")
        return ()

    if not data:
        return (0,)

    try:
        first_element_shape = _get_list_shape(data[0])
    except (ValueError, TypeError) as e:
        raise type(e)(f"Error processing element at index 0: {e}") from e

    for i, element in enumerate(data[1:], start=1):
        try:
            element_shape = _get_list_shape(element)
            if element_shape != first_element_shape:
                raise ValueError(
                    f"Inconsistent shape: Element at index 0 implies sub-shape {first_element_shape}, "
                    f"but element at index {i} has sub-shape {element_shape}."
                )
        except (ValueError, TypeError) as e:
            raise type(e)(f"Error processing element at index {i}: {e}") from e
    return (len(data),) + first_element_shape


def _calc_broadcast_shape(
    shape1: tuple[int, ...], shape2: tuple[int, ...]
) -> tuple[int, ...]:
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
            raise ValueError

    return tuple(result_shape_reversed[::-1])


def _calc_matmul_shape(
    shape1: tuple[int, ...], shape2: tuple[int, ...]
) -> tuple[int, ...]:
    ndim1 = len(shape1)
    ndim2 = len(shape2)

    if ndim1 == 0 or ndim2 == 0:
        # TODO: fallback to `multiply`
        raise ValueError(f"Use `mul` to deal with scalars, got shape {shape1} and {shape2}")

    # vector x vector
    if ndim1 == 1 and ndim2 == 1:
        # (k,) * (k,) -> float
        if shape1[0] != shape2[0]:
            raise ValueError(f"Missmatch of inner product vectors: {shape1} and {shape2}")
        return ()

    k1_dim = shape1[-1]
    k2_dim = shape2[-2] if ndim2 >= 2 else shape2[-1]

    if k1_dim != k2_dim:
        raise ValueError(f"Shape incompatibility for `matmul` between {shape1} and {shape2}")

    batch_shape1 = shape1[:-1] if ndim1 == 1 else shape1[:-2]
    batch_shape2 = shape2[:-1] if ndim2 == 1 else shape2[:-2]

    try:
        out_batch_shape = _calc_broadcast_shape(batch_shape1, batch_shape2)
    except ValueError:
        raise ValueError(f"Batch dimensions imcompatible in `matmul` for {shape1} and {shape2}")

    out_shape = tuple()


    if ndim1 >= 2 and ndim2 >= 2:
        # (m, k) * (k, n) -> (m, n)
        out_shape = (shape1[-2], shape2[-1])
    elif ndim1 >= 2:
        # (m, k) * (k,) -> (m,)
        out_shape = (shape1[-2],)
    elif ndim1 == 1 and ndim2 == 2:
        # (k,) * (k,n) -> (n,)
        out_shape = (shape2[-1],)
    else:
        raise RuntimeError("Internal error in shape inference with `matmul`")

    return out_batch_shape + out_shape


def _normalize_axis(
    axis: Optional[int | Sequence[int]], ndim: int
) -> tuple[int, ...]:
    """
    Normalizes axis argument for max, sum, ...
    """

    if axis is None:
        return tuple(range(ndim))  # all axis

    if ndim == 0:
        if axis:
            raise ValueError("Axis out of bounds")
        return tuple()

    if isinstance(axis, int):
        axis_tuple = (axis,)
    elif isinstance(axis, Sequence) and not isinstance(axis, (str, bytes)):
        axis_tuple = tuple(axis)
    else:
        raise TypeError(f"Invalid type for axis provided: {type(axis)}")

    # remove duplicates
    normalized_axes: set[int] = set()
    for ax in axis_tuple:
        if not isinstance(ax, int):
            raise TypeError(f"Invalid type for axis provided: {type(axis)}")

        if ax < -ndim or ax >= ndim:
            raise ValueError("Axis out of bounds")

        # neg to positive
        normalized_axes.add(ax % ndim)
    return tuple(sorted(list(normalized_axes)))


def _calc_reduction_shape(
    in_shape: tuple[int, ...],
    axis: Optional[int | Sequence[int]],
    keepdims: bool = False,
) -> tuple[int, ...]:
    ndim = len(in_shape)

    axis = _normalize_axis(axis, ndim)

    if not ndim:
        return tuple()

    if len(axis) == ndim:
        if keepdims:
            return (1,) * ndim
        else:
            return tuple()

    axis_set = set(axis)
    out_shape = []

    for i in range(ndim):
        if i in axis_set:
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(in_shape[i])

    return tuple(out_shape)

