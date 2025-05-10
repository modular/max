# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops that accumulate or reduce a tensor along an axis."""

from collections import Optional

from ..error import error


fn _reduce[
    op: StringLiteral
](v: Symbol, owned axis: Int, dtype: Optional[DType] = None) raises -> Symbol:
    var g = v.graph()
    var v_type = v.tensor_type()

    if axis < 0:
        axis += v_type.rank()
    if not 0 <= axis < v_type.rank():
        raise error("axis out of range")

    v_type.dims[axis] = 1
    if dtype:
        v_type.dtype = dtype._value_copy()

    return g.op(op, List[Symbol](v, g.scalar[DType.int64](axis)), v_type)


def mean(v: Symbol, axis: Int = -1) -> Symbol:
    """Reduces a symbolic tensor using a mean operation.

    Args:
        v: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension, ie. a value of -1 will compute
            the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the mean operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the `axis` dimension which will have size 1.
    """
    return _reduce["mo.mean"](v, axis)


def arg_max(v: Symbol, axis: Int = -1) -> Symbol:
    """Finds the index of the maximum value along a dimension.

    Args:
        v: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension, ie. a value of -1 will compute
            the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the arg_max operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the `axis` dimension which will have size 1.
    """
    return _reduce["mo.arg_max"](v, axis, dtype=DType.int64)
