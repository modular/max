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
# RUN: %mojo %s | FileCheck %s


from memory import DTypePointer

from utils import StringRef, StaticIntTuple


# CHECK-LABEL: test_print
fn test_print():
    print("== test_print")

    var a: SIMD[DType.float32, 2] = 5
    var b: SIMD[DType.float64, 4] = 6
    var c: SIMD[DType.index, 8] = 7

    # CHECK: False
    print(False)

    # CHECK: True
    print(True)

    # CHECK: [5.0, 5.0]
    print(a)

    # CHECK: [6.0, 6.0, 6.0, 6.0]
    print(b)

    # CHECK: [7, 7, 7, 7, 7, 7, 7, 7]
    print(c)

    # CHECK: Hello
    print("Hello")

    # CHECK: World
    print("World", flush=True)

    # CHECK: 4294967295
    print(UInt32(-1))

    # CHECK: 184467440737095516
    print(UInt64(-1))

    # CHECK: 0x16
    print(Scalar[DType.address](22))

    # CHECK: 0xdeadbeaf
    print(Scalar[DType.address](0xDEADBEAF))

    var hello: StringRef = "Hello,"
    var world: String = "world!"
    var f: Bool = False
    # CHECK: > Hello, world! 42 True False [5.0, 5.0] [7, 7, 7, 7, 7, 7, 7, 7]
    print(">", hello, world, 42, True, f, a, c)

    # CHECK: > 3.14000{{[0-9]+}} 99.90000{{[0-9]+}} -129.29018{{[0-9]+}} (1, 2, 3)
    var float32: Float32 = 99.9
    var float64: Float64 = -129.2901823
    print("> ", end="")
    print(3.14, float32, float64, StaticIntTuple[3](1, 2, 3), end="")
    print()

    # CHECK: > 9223372036854775806
    print(">", 9223372036854775806)

    var pi = 3.1415916535897743
    # CHECK: > 3.1415916535{{[0-9]+}}
    print(">", pi)
    var x = (pi - 3.141591) * 1e6
    # CHECK: > 0.6535{{[0-9]+}}
    print(">", x)

    # CHECK: Hello world
    print(String("Hello world"))

    # CHECK: 32768
    print((UInt16(32768)))
    # CHECK: 65535
    print((UInt16(65535)))
    # CHECK: -2
    print((Int16(-2)))

    # CHECK: 16646288086500911323
    print(UInt64(16646288086500911323))

    # https://github.com/modularml/mojo/issues/556
    # CHECK: [11562461410679940143, 16646288086500911323, 10285213230658275043, 6384245875588680899]
    print(
        SIMD[DType.uint64, 4](
            0xA0761D6478BD642F,
            0xE7037ED1A0B428DB,
            0x8EBC6AF09C88C6E3,
            0x589965CC75374CC3,
        )
    )

    # CHECK: [-943274556, -875902520, -808530484, -741158448]
    print(SIMD[DType.int32, 4](-943274556, -875902520, -808530484, -741158448))

    # CHECK: bad
    print(Error("bad"))


# CHECK-LABEL: test_print_end
fn test_print_end():
    print("== test_print_end")
    # CHECK: Hello World
    print("Hello", end=" World\n")


# CHECK-LABEL: test_print_sep
fn test_print_sep():
    print("== test_print_sep")

    # CHECK: a/b/c
    print("a", "b", "c", sep="/")

    # CHECK: a/1/2xx
    print("a", 1, 2, sep="/", end="xx\n")


# CHECK-LABEL: test_issue_20421
fn test_issue_20421():
    print("== test_issue_20421")
    var a = DTypePointer[DType.uint8]().alloc(16 * 64, alignment=64)
    for i in range(16 * 64):
        a[i] = i & 255
    var av16 = a.offset(128 + 64 + 4).bitcast[DType.int32]().load[width=4]()
    # CHECK: [-943274556, -875902520, -808530484, -741158448]
    print(av16)
    a.free()


fn main():
    test_print()
    test_print_end()
    test_print_sep()
    test_issue_20421()
