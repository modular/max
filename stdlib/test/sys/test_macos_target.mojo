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
#
# This file is only run on macos targets.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: darwin
# RUN: %mojo %s | FileCheck %s


from sys.info import (
    is_big_endian,
    is_little_endian,
    os_is_linux,
    os_is_macos,
    os_is_windows,
)


# CHECK-LABEL: test_os_query
fn test_os_query():
    print("== test_os_query")

    # CHECK: True
    print(os_is_macos())

    # CHECK: False
    print(os_is_linux())

    # CHECK: False
    print(os_is_windows())

    # The mac systems are either arm64 or intel, so they are always little
    # endian at the moment.

    # CHECK: True
    print(is_little_endian())

    # CHECK: False
    print(is_big_endian())


fn main():
    test_os_query()
