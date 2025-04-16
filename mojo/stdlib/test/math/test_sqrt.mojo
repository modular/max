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
# RUN: %mojo %s

from math import sqrt
from testing import assert_equal


def main():
    assert_equal(sqrt(-1), 0)
    assert_equal(sqrt(0), 0)
    assert_equal(sqrt(1), 1)
    assert_equal(sqrt(2**34 - 1), 2**17 - 1)
    assert_equal(sqrt(2**34), 2**17)
    assert_equal(sqrt(10**16), 10**8)
    assert_equal(sqrt(Int.MAX), 3037000499)
