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
from compiler import register
from math import iota
from memory import Span
from utils.index import IndexList

from max.tensor import ManagedTensorSlice


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
    ](
        out_vals: ManagedTensorSlice[type, rank],
        out_idxs: ManagedTensorSlice[DType.int64, rank],
        in_vals: ManagedTensorSlice[type, rank],
        k: Scalar,
        axis: Scalar,
    ):
        var shape = in_vals.shape()
        alias idx = IndexList[rank]

        @parameter
        fn process_rows(start_row: Int, end_row: Int):
            # Total amount of elements to find top k for
            var els = shape[axis]

            for row_idx in range(start_row, end_row):
                var offset = (row_idx * els)
                iota(out_idxs.unsafe_ptr() + offset, shape[1])

                @parameter
                fn val_greater_than(lhs: Int64, rhs: Int64) -> Bool:
                    return (
                        in_vals[idx(row_idx, Int(lhs))]
                        > in_vals[idx(row_idx, Int(rhs))]
                    )

                sort[val_greater_than](
                    Span(out_idxs.unsafe_ptr() + offset, els)
                )

                for i in range(els):
                    var sorted_idx = Int(out_idxs[idx(row_idx, i)])
                    out_vals[idx(row_idx, i)] = in_vals[
                        idx(row_idx, sorted_idx)
                    ]

        # Set grain size to 1 to put each batch in a separate task
        parallelize_over_rows[process_rows](shape, Int(axis), grain_size=1)

    @staticmethod
    fn shape[
        axis_type: DType
    ](
        in_vals: ManagedTensorSlice,
        k: Scalar[axis_type],
        axis: Scalar[axis_type],
    ) raises -> IndexList[in_vals.rank]:
        if 0 > Int(axis) >= in_vals.rank:
            raise Error("[top-k] axis must be within [0, rank]")
        if 0 > Int(k) > in_vals.shape()[0]:
            raise Error("[top-k] k must be within [0, input_shape[axis]]")
        var shape = in_vals.shape()
        shape[Int(axis)] = Int(k)
        return shape
