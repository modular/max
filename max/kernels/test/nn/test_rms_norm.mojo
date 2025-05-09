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

from math import ceildiv, sqrt
from sys import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from nn.normalization import *
from testing import assert_almost_equal
from utils.index import Index, IndexList


fn compute_rms[
    type: DType
](data: NDBuffer[type, 1], size: Int, eps: Scalar[type]) -> Scalar[type]:
    var sum_of_squares = Scalar[type]()
    for i in range(size):
        sum_of_squares += data[i] * data[i]
    return sqrt((sum_of_squares / len(data)) + eps).cast[type]()


fn run_rms_norm_cpu[
    type: DType, rank: Int
](shape: IndexList[rank], rtol: Float64 = 0.001) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var input_ptr = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var output_ptr = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_ptr = UnsafePointer[Scalar[type]].alloc(cols)

    for i in range(rows * cols):
        input_ptr[i] = Scalar[type](i)

    for i in range(cols):
        gamma_ptr[i] = ((i + cols) / cols).cast[type]()

    var param_shape = Index(cols)

    var input_buf = NDBuffer[type, rank](input_ptr, shape)
    var output_buf = NDBuffer[type, rank](output_ptr, shape)
    var gamma = NDBuffer[type, 1](gamma_ptr, param_shape)
    var epsilon = Scalar[type](0.0001)
    var weight_offset = Scalar[type](0.0)

    @__copy_capture(input_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, width]:
        return input_buf.load[width=width](rebind[IndexList[rank]](idx))

    @always_inline
    @__copy_capture(output_buf)
    @parameter
    fn identity_output_fn[
        width: Int
    ](idx: IndexList[rank], val: SIMD[type, width]) -> None:
        output_buf.store(idx, val)

    rms_norm_cpu[input_fn, identity_output_fn, multiply_before_cast=True](
        shape,
        gamma,
        epsilon,
        weight_offset,
    )

    for r in range(rows):
        var vec = NDBuffer[type, 1](input_ptr + r * cols, cols)
        var rms_ref = compute_rms(vec, cols, epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = (input_ptr[idx] / rms_ref) * (
                gamma_ptr[c] + weight_offset
            )
            assert_almost_equal(val, output_ptr[idx], rtol=rtol)

    input_ptr.free()
    output_ptr.free()
    gamma_ptr.free()


def main():
    run_rms_norm_cpu[DType.float32](Index(2, 5))
    run_rms_norm_cpu[DType.float32](Index(2, 55))
    run_rms_norm_cpu[DType.float32](Index(7, 557))
    run_rms_norm_cpu[DType.float32](Index(2, 8191))
    run_rms_norm_cpu[DType.float32](Index(2, 8192))
    run_rms_norm_cpu[DType.float32](Index(2, 16384))
    run_rms_norm_cpu[DType.float32](Index(2, 16385))

    # variable rank
    run_rms_norm_cpu[DType.float32](Index(0))
    run_rms_norm_cpu[DType.float32](Index(5))
    run_rms_norm_cpu[DType.float32](Index(3, 4, 10, 20, 8))
    run_rms_norm_cpu[DType.float32](Index(1, 5, 6, 10, 128))
