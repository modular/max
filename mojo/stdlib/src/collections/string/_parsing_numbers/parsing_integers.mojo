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

from .constants import CONTAINER_SIZE, MAXIMUM_UINT64_AS_STRING
from memory import memcpy, memcmp


fn standardize_string_slice(
    x: StringSlice,
) -> InlineArray[UInt8, CONTAINER_SIZE]:
    """Put the input string in an inline array, aligned to the right and padded with "0" on the left.
    """
    standardized_x = InlineArray[UInt8, CONTAINER_SIZE](ord("0"))
    memcpy(
        dest=(standardized_x.unsafe_ptr() + CONTAINER_SIZE - len(x)).bitcast[Int8](),
        src=x.unsafe_ptr().bitcast[Int8](),
        count=len(x),
    )
    return standardized_x


# The idea is to end up with a InlineArray of size
# 24, which is enough to store the largest integer
# that can be represented in unsigned 64 bits (size 20), and
# is also SIMD friendly because divisible by 8, 4, 2, 1.
# This 24 could be computed at compile time and adapted
# to the simd width and the base, but Mojo's compile-time
# computation is not yet powerful enough yet.
# For now we focus on base 10.
fn to_integer(x: String) raises -> UInt64:
    return to_integer(x.as_string_slice())


fn to_integer(x: StringSlice) raises -> UInt64:
    """The input does not need to be padded with "0" on the left.
    The function returns the integer value represented by the input string.
    """
    if len(x) > len(MAXIMUM_UINT64_AS_STRING):
        raise Error("The string size too big. '" + String(x) + "'")
    return to_integer(standardize_string_slice(x))


fn to_integer(
    standardized_x: InlineArray[UInt8, CONTAINER_SIZE]
) raises -> UInt64:
    """Takes a inline array containing the ASCII representation of a number.
    It must be padded with "0" on the left. Using an InlineArray makes
    this SIMD friendly.

    We assume there are no leading or trailing whitespaces, no sign, no underscore.

    The function returns the integer value represented by the input string.

    "000000000048642165487456" -> 48642165487456
    """

    # This could be done with simd if we see it's a bottleneck.
    for i in range(CONTAINER_SIZE):
        if not (UInt8(ord("0")) <= standardized_x[i] <= UInt8(ord("9"))):
            # We make a string out of this number. +1 for the null terminator.
            number_as_string = String()
            for j in range(CONTAINER_SIZE):
                number_as_string += chr(Int(standardized_x[j]))
            raise Error(
                "Invalid character(s) in the number: '"
                + String(number_as_string^)
                + "'"
            )

    # 24 is not divisible by 16, so we stop at 8. Later on,
    # when we have better compile-time computation, we can
    # change 24 to be adapted to the simd width.
    alias simd_width = min(sys.simdwidthof[DType.uint64](), 8)

    accumulator = SIMD[DType.uint64, simd_width](0)

    # We use memcmp to check that the number is not too large.
    alias max_standardized_x = String(UInt64.MAX).rjust(CONTAINER_SIZE, "0")
    if (
        memcmp(
            standardized_x.unsafe_ptr(),
            max_standardized_x.unsafe_ptr(),
            count=CONTAINER_SIZE,
        )
        == 1
    ):
        raise Error("The string is too large to be converted to an integer. '")

    # actual conversion
    alias vector_with_exponents = get_vector_with_exponents()

    @parameter
    for i in range(CONTAINER_SIZE // simd_width):
        ascii_vector = (standardized_x.unsafe_ptr() + i * simd_width).load[
            width=simd_width
        ]()
        as_digits = ascii_vector - SIMD[DType.uint8, simd_width](ord("0"))
        as_digits_index = as_digits.cast[DType.uint64]()
        alias vector_slice = (
            vector_with_exponents.unsafe_ptr() + i * simd_width
        ).load[width=simd_width]()
        accumulator += as_digits_index * vector_slice
    return Int(accumulator.reduce_add())


fn get_vector_with_exponents() -> InlineArray[UInt64, CONTAINER_SIZE]:
    """Returns (0, 0, 0, 0, 10**19, 10**18, 10**17, ..., 10, 1)."""
    result = InlineArray[UInt64, CONTAINER_SIZE](0)
    for i in range(4, CONTAINER_SIZE):
        result[i] = 10**(CONTAINER_SIZE - i - 1)
    return result
