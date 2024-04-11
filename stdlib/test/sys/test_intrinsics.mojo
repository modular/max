# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.intrinsics import (
    compressed_store,
    masked_load,
    masked_store,
    strided_load,
    strided_store,
)

from memory.unsafe import DTypePointer
from testing import assert_equal


fn test_strided_load() raises:
    alias size = 16
    var vector = DTypePointer[DType.float32]().alloc(size)

    for i in range(size):
        vector[i] = i

    var s = strided_load[DType.float32, 4](vector, 4)
    assert_equal(s, SIMD[DType.float32, 4](0, 4, 8, 12))

    vector.free()


# CHECK-LABEL: test_strided_store
fn test_strided_store():
    print("== test_strided_store")

    alias size = 8
    var vector = DTypePointer[DType.float32]().alloc(size)
    memset_zero(vector, size)

    strided_store(SIMD[DType.float32, 4](99, 12, 23, 56), vector, 2)
    # CHECK: 99.0
    # CHECK: 0.0
    # CHECK: 12.0
    # CHECK: 0.0
    # CHECK: 23.0
    # CHECK: 0.0
    # CHECK: 56.0
    # CHECK: 0.0
    for i in range(size):
        print(vector[i])
    vector.free()


def main():
    test_strided_load()
    test_strided_store()
