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
"""Implements compile time constraints.

These are Mojo built-ins, so you don't need to import them.
"""
from collections.string.string_slice import StaticString
from builtin.string_literal import get_string_literal_slice2


@always_inline("nodebug")
fn constrained[cond: Bool, msg: StaticString, *extra: StaticString]():
    """Compile time checks that the condition is true.

    The `constrained` is similar to `static_assert` in C++ and is used to
    introduce constraints on the enclosing function. In Mojo, the assert places
    a constraint on the function. The message is displayed when the assertion
    fails.

    Parameters:
        cond: The bool value to assert.
        msg: The message to display on failure.
        extra: Additional messages to concatenate to msg.

    Examples:

    ```mojo
    from sys.info import num_physical_cores

    def main():
        alias cores_to_use = 2
        multicore_check[cores_to_use]()

    def multicore_check[cores: Int]():
        constrained[
            cores <= num_physical_cores(),
            "build failed: not enough cores"
        ]()
        constrained[
            cores >= 2,
            "at least two cores are required"
        ]()
    ```
    """
    __mlir_op.`kgen.param.assert`[
        cond = cond.__mlir_i1__(),
        message = get_string_literal_slice2[msg, extra]().value,
    ]()


@always_inline("nodebug")
fn constrained[cond: Bool]():
    """Compile time checks that the condition is true.

    The `constrained` is similar to `static_assert` in C++ and is used to
    introduce constraints on the enclosing function. In Mojo, the assert places
    a constraint on the function.

    Parameters:
        cond: The bool value to assert.

    Examples:

    ```mojo
    from sys.info import num_physical_cores

    def main():
        alias cores_to_use = 2
        multicore_check[cores_to_use]()

    def multicore_check[cores: Int]():
        constrained[cores <= num_physical_cores()]()
        constrained[cores >= 2]()
    ```
    """
    constrained[cond, "param assertion failed"]()
