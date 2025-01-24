# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import compiler
from algorithm.functional import parallelize_over_rows
from algorithm.reduction import _get_nd_indices_from_flat_index
from compiler import StaticTensorSpec
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from memory import UnsafePointer, Span
from utils.index import IndexList
from tensor_utils import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr
from math import iota

alias ScalarTensor = ManagedTensorSlice[rank=1]


@always_inline
fn managed_tensor_slice_to_ndbuffer_with_spec[
    spec: StaticTensorSpec
](tensor: ManagedTensorSlice[spec.type, spec.rank]) -> NDBuffer[
    spec.type,
    spec.rank,
    spec.shape,
    spec.strides,
    alignment = spec.alignment,
    address_space = spec.address_space,
    exclusive = spec.exclusive,
]:
    var ptr = tensor._ptr.address_space_cast[spec.address_space]()
    return NDBuffer[
        spec.type,
        spec.rank,
        spec.shape,
        spec.strides,
        alignment = spec.alignment,
        address_space = spec.address_space,
        exclusive = spec.exclusive,
    ](ptr, tensor.shape(), tensor._runtime_strides)


@always_inline
fn top_k_shape_impl[
    type: DType,
    rank: Int,
    axis_type: DType, //,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    k_buf: NDBuffer[axis_type, 1],
    axis_buf: NDBuffer[axis_type, 1],
) raises -> IndexList[rank]:
    """
    Compute the output shape of a top/bottom k operation.

    Parameters:
        type: Data type of the input buffer.
        rank: Rank of the input.
        axis_type: Type of the axis and K arguments.
        single_thread_blocking_override: If this function can block.

    Args:
        input: The input tensor.
        k_buf: The K value in a tensor.
        axis_buf: The axis value in a tensor.

    Returns:
        The output shape.
    """
    var axis = Int(axis_buf[0])
    var k = Int(k_buf[0])

    if axis < 0 or axis >= rank:
        raise Error("[top/bottom-k] axis must be within [0, rank]")
    if k < 0 or k > input.get_shape()[axis]:
        raise Error("[top/bottom-k] k must be within [0, input_shape[axis]]")

    var shape = input.get_shape()
    shape[axis] = k

    return shape


fn top_k[
    rank: Int, type: DType, //
](
    input: NDBuffer[type, rank],
    k: Int,
    axis: Int,
    largest: Bool,
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[DType.int64, rank],
    sorted: Bool = True,
    grain_size: Int = 1000,
):
    """
    Implementation of the Top K algorithm. Returns the top or bottom K elements
    and their index along a specified axis.

    Parameters:
        rank: Rank of the input.
        type: Data type of the input buffer.

    Args:
        input: The input tensor.
        k: Represents the K largest/smallest value.
        axis: On which axis it should operate.
        largest: If true, acts like top K. Otherwise, bottom K.
        out_vals: Output values.
        out_idxs: Output indices.
        sorted: Indicates if the top/bottom K elements are in (stable) sorted order.
        grain_size: The minimum number of elements to warrant using an additional thread.
    """
    var shape = input.get_shape()

    @__copy_capture(shape)
    @parameter
    fn process_rows(start_row: Int, end_row: Int):
        # Allocate the index list without initializing its elements.
        var idxs = List(
            ptr=UnsafePointer[Int64].alloc(shape[axis]),
            length=shape[axis],
            capacity=shape[axis],
        )

        for row_idx in range(start_row, end_row):
            var indices = _get_nd_indices_from_flat_index(row_idx, shape, axis)
            iota(idxs)

            @parameter
            @always_inline
            fn indices_to_val(idx: Int64) -> Scalar[type]:
                indices[axis] = Int(idx)
                return input[indices]

            if largest:

                @parameter
                @always_inline
                fn _val_greater_than(lhs: Int64, rhs: Int64) -> Bool:
                    return indices_to_val(lhs) > indices_to_val(rhs)

                if sorted:
                    sort[_val_greater_than](idxs)
                else:
                    _ = partition[_val_greater_than](idxs, k)
            else:

                @parameter
                @always_inline
                fn _val_less_than(lhs: Int64, rhs: Int64) -> Bool:
                    return indices_to_val(lhs) < indices_to_val(rhs)

                if sorted:
                    sort[_val_less_than](idxs)
                else:
                    _ = partition[_val_less_than](idxs, k)

            if sorted:
                # for duplicate vals, the smaller index needs to appear first
                # _quicksort is not stable, so do another pass to enforce this
                # could use a stable sorting algorithm but the complexity is O(n*log(n)*log(n))
                # this is also what tensorflow and PT do:
                # https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/kernels/topk_op.cc#L171-L172
                var i = 0
                while i < shape[axis] - 1:
                    indices[axis] = Int(idxs[i])
                    var curr = input[indices]
                    var num_equal = 1
                    for j in range(i + 1, shape[axis]):
                        indices[axis] = Int(idxs[j])
                        var next = input[indices]
                        if curr != next:
                            break
                        num_equal += 1
                    if num_equal > 1:
                        var ptr = idxs.data + i
                        sort(
                            Span[idxs.T, __origin_of(idxs)](
                                ptr=ptr, length=num_equal
                            )
                        )
                    i += num_equal

            for i in range(k):
                indices[axis] = Int(idxs[i])
                var val = input[indices]
                indices[axis] = i
                out_vals[indices] = val
                out_idxs[indices] = idxs[i]

    parallelize_over_rows[process_rows](shape, axis, grain_size)


@compiler.register("top_k_custom", num_dps_outputs=2)
struct TopK:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        values: ManagedTensorSlice[type=type, rank=rank],
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[type = DType.bool],
    ):
        top_k(
            managed_tensor_slice_to_ndbuffer_with_spec[
                compiler.specsof[input.type, input.rank]("input")
            ](input),
            Int(k),
            Int(axis),
            True,
            managed_tensor_slice_to_ndbuffer_with_spec[
                compiler.specsof[values.type, values.rank]("values")
            ](values),
            managed_tensor_slice_to_ndbuffer_with_spec[
                compiler.specsof[indices.type, indices.rank]("indices")
            ](indices),
            sorted,
        )

    @staticmethod
    fn shape[
        axis_type: DType
    ](
        input: ManagedTensorSlice,
        k: ScalarTensor[axis_type],
        axis: ScalarTensor[axis_type],
        sorted: ScalarTensor[type = DType.bool],
    ) raises -> IndexList[input.rank]:
        return top_k_shape_impl[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer_with_spec[
                compiler.specsof[input.type, input.rank]("input")
            ](input),
            managed_tensor_slice_to_ndbuffer_with_spec[
                compiler.specsof[k.type, k.rank]("k")
            ](k),
            managed_tensor_slice_to_ndbuffer_with_spec[
                compiler.specsof[axis.type, axis.rank]("axis")
            ](axis),
        )
