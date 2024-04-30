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


fn test_list() raises:
    assert_equal(len([1, 2.0, 3.14, [-1, -2]]), 4)


fn test_variadic_list() raises:
    @parameter
    fn print_list(*nums: Int) raises:
        assert_equal(nums[0], 5)
        assert_equal(nums[1], 8)
        assert_equal(nums[2], 6)

        assert_equal(len(nums), 3)

    print_list(5, 8, 6)


def main():
    test_list()
    test_variadic_list()
