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

from algorithm import parallelize_over_rows
from bit import log2_floor
from compiler import register
from gpu import barrier, shuffle_down, WARP_SIZE
from gpu.memory import AddressSpace, external_memory
from math import iota
from max.tensor import ManagedTensorSlice
from memory import Span
from sys import alignof, sizeof
from utils.index import IndexList
from utils.numerics import min_or_neg_inf


@register("top_k_custom", num_dps_outputs=2)
struct TopK:
    """Registers the `top_k_custom` op, allowing python to use it from the `max`
    package. This is a simplified version without bottom_k and sorting options,
    or fused sampling. The purpose is to demonstrate concisely how you can
    implement your own custom ops in Mojo that can be called from Python. MAX
    has the "mo.top_k" op which is feature complete.
    """

    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        //,  # Forces the previous two params to be inferred from the args
        K: Int,
        target: StringLiteral,
    ](
        out_vals: ManagedTensorSlice[type, rank],
        out_idxs: ManagedTensorSlice[DType.int32, rank],
        in_vals: ManagedTensorSlice[type, rank],
        ctx: MojoCallContextPtr,
    ) raises:
        constrained[rank == 2, "rank must be 2"]()
        var shape = in_vals.shape()

        @parameter
        if target == "cpu":
            print("Running on CPU")

            @parameter
            fn process_rows(start_row: Int, end_row: Int):
                # Total amount of elements to find top k for

                for row_idx in range(start_row, end_row):
                    var offset = (row_idx * K)
                    iota(out_idxs.unsafe_ptr() + offset, K)

                    @parameter
                    fn val_greater_than(lhs: Int32, rhs: Int32) -> Bool:
                        return (
                            in_vals[row_idx, Int(lhs)]
                            > in_vals[row_idx, Int(rhs)]
                        )

                    sort[val_greater_than](
                        Span(out_idxs.unsafe_ptr() + offset, K)
                    )

                    for i in range(K):
                        var sorted_idx = Int(out_idxs[row_idx, i])
                        out_vals[row_idx, i] = in_vals[row_idx, sorted_idx]

            # Set grain size to 1 to put each batch in a separate task
            parallelize_over_rows[process_rows](shape, 1, grain_size=1)
        else:
            var dev_ctx = ctx.get_device_context()
            print("Running on GPU:", dev_ctx.name())
            # This is a simplified version of top_k that only works for K being
            # under the warp size. The MAX "mo.top_k" op supports any K and
            # does another reduction after each warp has reduced it's values.
            if K > WARP_SIZE:
                raise Error(
                    "[top_k_custom] K=",
                    K,
                    " but must be less than or equal to WARP_SIZE=",
                    WARP_SIZE,
                )

            dev_ctx.enqueue_function(
                dev_ctx.compile_function[top_k_gpu[type, rank, K]](),
                in_vals,
                out_vals,
                out_idxs,
                grid_dim=shape[0],  # One block per batch
                block_dim=K,
                shared_mem_bytes=K * sizeof[TopKElement[type]](),
            )


@value
@register_passable("trivial")
struct TopKElement[T: DType]:
    """Stores the value with it's index."""

    var idx: Int32
    var val: Scalar[T]


fn top_k_gpu[
    T: DType,
    rank: Int,
    K: Int,
](
    in_vals: ManagedTensorSlice[T, rank],
    out_vals: ManagedTensorSlice[T, rank],
    out_idxs: ManagedTensorSlice[DType.int32, rank],
):
    var bid = block_idx.x
    var tid = thread_idx.x

    # Get a pointer to shared memory in this block for the indices and values
    var top_k_sram = external_memory[
        TopKElement[T],
        address_space = AddressSpace.SHARED,
        alignment = alignof[TopKElement[T]](),
    ]()

    # Each thread puts it's corresponding index and value into shared memory
    top_k_sram[tid] = TopKElement[T](tid, in_vals[bid, tid])
    # Finish packing the values across threads in this block before the next step
    barrier()

    for i in range(K):
        var reduced = top_k_sram[tid]
        alias limit = log2_floor(K)

        # TODO(KERN-1544): allow gpu.shuffle.warp_max to be used with index and value
        @parameter
        for j in reversed(range(limit)):
            alias offset = 1 << j
            # Parallel reduction using warp shuffle. Each thread gets a value
            # from a thread 'offset' positions higher, keeping the larger value.
            var shuffled = TopKElement[T](
                shuffle_down(reduced.idx, offset),
                shuffle_down(reduced.val, offset),
            )
            reduced = reduced if reduced.val > shuffled.val else shuffled

        # Wait for all threads to finish reducing their values for this index
        barrier()

        # Thread 0 now has the reduced max value for this index in the batch
        if tid == 0:
            # Store the reduced top_k index and value in global memory for the CPU
            out_vals[bid, i] = reduced.val
            out_idxs[bid, i] = reduced.idx

            # Remove the found maximum from consideration in the next iteration
            top_k_sram[reduced.idx % block_dim.x].val = min_or_neg_inf[T]()
