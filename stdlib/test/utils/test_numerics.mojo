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
# RUN: %mojo %s

from sys.info import has_neon
from testing import assert_equal, assert_true, assert_false
from utils._numerics import FPUtils, inf, isinf

alias FPU64 = FPUtils[DType.float64]


fn test_numerics() raises:
    assert_equal(FPUtils[DType.float32].mantissa_width(), 23)

    assert_equal(FPUtils[DType.float64].mantissa_width(), 52)

    assert_equal(FPUtils[DType.float32].exponent_bias(), 127)

    assert_equal(FPUtils[DType.float64].exponent_bias(), 1023)

    assert_equal(FPU64.get_exponent(FPU64.set_exponent(1, 2)), 2)
    assert_equal(FPU64.get_mantissa(FPU64.set_mantissa(1, 3)), 3)
    assert_equal(FPU64.get_exponent(FPU64.set_exponent(-1, 4)), 4)
    assert_equal(FPU64.get_mantissa(FPU64.set_mantissa(-1, 5)), 5)
    assert_true(FPU64.get_sign(FPU64.set_sign(0, True)))
    assert_false(FPU64.get_sign(FPU64.set_sign(0, False)))
    assert_true(FPU64.get_sign(FPU64.set_sign(-0, True)))
    assert_false(FPU64.get_sign(FPU64.set_sign(-0, False)))
    assert_false(FPU64.get_sign(1))
    assert_true(FPU64.get_sign(-1))
    assert_false(FPU64.get_sign(FPU64.pack(False, 6, 12)))
    assert_equal(FPU64.get_exponent(FPU64.pack(False, 6, 12)), 6)
    assert_equal(FPU64.get_mantissa(FPU64.pack(False, 6, 12)), 12)
    assert_true(FPU64.get_sign(FPU64.pack(True, 6, 12)))
    assert_equal(FPU64.get_exponent(FPU64.pack(True, 6, 12)), 6)
    assert_equal(FPU64.get_mantissa(FPU64.pack(True, 6, 12)), 12)


fn test_inf() raises:
    @parameter
    fn _test_inf[type: DType]() raises:
        var val = inf[type]()
        var msg = "`test_inf` failed for `type == " + str(type) + "`"
        assert_true((val > 0.0) & isinf(val), msg=msg)

    @parameter
    if not has_neon():
        # "bf16 is not supported for ARM architectures"
        _test_inf[DType.bfloat16]()

    _test_inf[DType.float16]()
    _test_inf[DType.float32]()
    _test_inf[DType.float64]()


def main():
    test_numerics()
    test_inf()
