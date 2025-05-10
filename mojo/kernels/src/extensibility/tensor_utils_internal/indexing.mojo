# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor_internal import StaticTensorSpec
from utils import IndexList
from collections import InlineArray


@always_inline
fn _dot_prod[rank: Int](x: IndexList[rank], y: IndexList[rank]) -> Int:
    var offset = 0

    @parameter
    for i in range(rank):
        offset += x[i] * y[i]
    return offset


@always_inline
fn _slice_to_tuple[
    func: fn (Slice) capturing [_] -> Int, rank: Int
](slices: InlineArray[Slice, rank]) -> IndexList[rank]:
    """Takes a tuple of `Slice`s and returns a tuple of Ints.
    `func` is used to extract the appropriate field (i.e. start, stop or end)
    of the Slice.
    """
    var tuple = IndexList[rank]()

    @parameter
    for i in range(rank):
        tuple[i] = func(slices[i])
    return tuple


@always_inline
fn _row_major_strides[
    type: DType, rank: Int
](spec: StaticTensorSpec[type, rank]) -> IndexList[rank]:
    var offset = 1
    var strides = IndexList[rank]()

    @parameter
    for i in range(rank - 1, -1, -1):
        strides[i] = offset
        offset *= spec.shape[i]
    return strides
