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

from testing import assert_equal


fn test_int() raises:
    assert_equal(3, 3)
    assert_equal(3 + 3, 6)
    assert_equal(4 - 1, 3)
    assert_equal(6 - 1, 5)


fn test_floordiv() raises:
    assert_equal(2 // 2, 1)
    assert_equal(2 // 3, 0)
    assert_equal(2 // -2, -1)
    assert_equal(99 // -2, -50)


fn test_mod() raises:
    assert_equal(99 % 1, 0)
    assert_equal(99 % 3, 0)
    assert_equal(99 % -2, -1)
    assert_equal(99 % 8, 3)
    assert_equal(99 % -8, -5)
    assert_equal(2 % -1, 0)
    assert_equal(2 % -2, 0)
    assert_equal(3 % -2, -1)
    assert_equal(-3 % 2, 1)


fn test_bit_width() raises:
    assert_equal((0)._bit_width(), 1)
    assert_equal((-1)._bit_width(), 1)
    assert_equal((255)._bit_width(), 9)
    assert_equal((-256)._bit_width(), 9)


def main():
    test_int()
    test_floordiv()
    test_mod()
    test_bit_width()
