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

"""
We make use of the following papers for the implementation, note that there
are some small differences.

Wojciech Muła, Daniel Lemire, Base64 encoding and decoding at almost the
speed of a memory copy, Software: Practice and Experience 50 (2), 2020.
https://arxiv.org/abs/1910.05109

Wojciech Muła, Daniel Lemire, Faster Base64 Encoding and Decoding using AVX2
Instructions, ACM Transactions on the Web 12 (3), 2018.
https://arxiv.org/abs/1704.00605
"""

from builtin.simd import _sub_with_saturation
from collections import InlineArray
from math.math import _iota
from memory import memcpy, bitcast, UnsafePointer
from utils import IndexList

alias Bytes = SIMD[DType.uint8, _]


# |                |---- byte 2 ----|---- byte 1 ----|---- byte 0 ----|
# |                |c₁c₀d₅d₄d₃d₂d₁d₀|b₃b₂b₁b₀c₅c₄c₃c₂|a₅a₄a₃a₂a₁a₀b₅b₄|
# <----------------|----------------|----------------|----------------|
# |31 . . . . . .24|23 . . . . . .16|15 . . . . . .08| 7 6 5 4 3 2 1 0|
# |                                                                   |
# |---- byte 1 ----|---- byte 2 ----|---- byte 0 ----|---- byte 1 ----|
# |b₃b₂b₁b₀c₅c₄c₃c₂|c₁c₀d₅d₄d₃d₂d₁d₀|a₅a₄a₃a₂a₁a₀b₅b₄|b₃b₂b₁b₀c₅c₄c₃c₂|
# |        -------------____________ ------------_____________        |
# |        [     C     ][     D    ] [    A     ][     B     ]        |
# |                                                                   |
# |--- ascii(d) ---|--- ascii(c) ---|--- ascii(b) ---|--- ascii(a) ---|
# |. . d₅d₄d₃d₂d₁d₀|. . c₅c₄c₃c₂c₁c₀|. . b₅b₄b₃b₂b₁b₀|. . a₅a₄a₃a₂a₁a₀|
fn _6bit_to_byte[width: Int](input: Bytes[width]) -> Bytes[width]:
    constrained[width in [4, 8, 16, 32, 64], "width must be between 4 and 64"]()

    fn indices() -> IndexList[width]:
        alias perm = List(1, 0, 2, 1)
        var res = IndexList[width]()
        for i in range(width // 4):
            for j in range(4):
                res[4 * i + j] = 3 * i + perm[j]
        return res

    @always_inline
    fn combine[
        mask: Bytes[4], shift: Int
    ](shuffled: Bytes[width]) -> Bytes[width]:
        var `6bit` = shuffled & _repeat_until[width](mask)
        return _rshift_bits_in_u16[shift](`6bit`)

    var shuffled = input.shuffle[mask = indices()]()
    var a = combine[
        Bytes[4](0b0000_0000, 0b1111_1100, 0b0000_0000, 0b0000_0000), 10
    ](shuffled)
    var b = combine[
        Bytes[4](0b1111_0000, 0b0000_0011, 0b0000_0000, 0b0000_0000), -4
    ](shuffled)
    var c = combine[
        Bytes[4](0b0000_0000, 0b0000_0000, 0b1100_0000, 0b0000_1111), 6
    ](shuffled)
    var d = combine[
        Bytes[4](0b0000_0000, 0b0000_0000, 0b0011_1111, 0b0000_0000), 8
    ](shuffled)
    return a | b | c | d


# | 6-bit Value | ASCII Range | Target index | Offset (6-bit to ASCII) |
# |-------------|-------------|--------------|-------------------------|
# |  0 ... 25   | A ... Z     | 13           | 65                      |
# | 26 ... 51   | a ... z     |  0           | 71                      |
# | 52 ... 61   | 0 ... 9     |  1 ... 10    | -4                      |
# | 62          | +           | 11           | -19                     |
# | 63          | /           | 12           | -16                     |
# fmt: off
alias UNUSED = 0
alias OFFSETS = Bytes[16](
    71,                                     # a ... z
    -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, # 0 ... 9
    -19,                                    # +
    -16,                                    # /
    65,                                     # A ... Z
    UNUSED, UNUSED
)
alias END_FIRST_RANGE = 25
alias END_SECOND_RANGE = 51
# fmt: on


fn _to_b64_ascii[width: Int, //](input: Bytes[width]) -> Bytes[width]:
    var abcd = _6bit_to_byte(input)
    var target_indices = _sub_with_saturation(abcd, END_SECOND_RANGE)
    var offset_indices = (abcd <= END_FIRST_RANGE).select(13, target_indices)
    return abcd + OFFSETS._dynamic_shuffle(offset_indices)


fn _get_number_of_bytes_to_store_from_number_of_bytes_to_load[
    max_size: Int
](nb_of_elements_to_load: Int) -> Int:
    alias table = _ceildiv_u8(_iota[DType.uint8, max_size](), 3) * 4
    return int(table[nb_of_elements_to_load])


fn _get_number_of_bytes_to_store_from_number_of_bytes_to_load_without_equal_sign[
    max_size: Int
](nb_of_elements_to_load: Int) -> Int:
    alias table = _ceildiv_u8(_iota[DType.uint8, max_size]() * 8, 6)
    return int(table[nb_of_elements_to_load])


# TODO: Use Span instead of List as input when Span is easier to use
@no_inline
fn b64encode_with_buffers(
    input_bytes: List[UInt8, _], inout result: List[UInt8, _]
):
    alias simd_width = sys.simdbytewidth()
    alias input_simd_width = simd_width * 3 // 4
    alias equal_vector = SIMD[DType.uint8, simd_width](ord("="))

    var input_bytes_len = len(input_bytes)

    var input_index = 0

    # Main loop
    while input_index + simd_width <= input_bytes_len:
        var start_of_input_chunk = input_bytes.unsafe_ptr() + input_index

        var input_vector = start_of_input_chunk.load[width=simd_width]()

        result_vector = _to_b64_ascii(input_vector)

        (result.unsafe_ptr() + len(result)).store(result_vector)

        result.size += simd_width
        input_index += input_simd_width

    # We handle the last 0, 1 or 2 chunks
    while input_index < input_bytes_len:
        var start_of_input_chunk = input_bytes.unsafe_ptr() + input_index
        var nb_of_elements_to_load = min(
            input_simd_width, input_bytes_len - input_index
        )

        # We don't want to read past the input buffer
        var input_vector = load_simd[simd_width](
            start_of_input_chunk,
            nb_of_elements_to_load,
        )

        result_vector = _to_b64_ascii(input_vector)

        # We place the '=' where needed
        var non_equal_chars_number = _get_number_of_bytes_to_store_from_number_of_bytes_to_load_without_equal_sign[
            simd_width
        ](
            nb_of_elements_to_load
        )
        var equal_mask = _iota[
            DType.uint8, simd_width
        ]() < non_equal_chars_number

        var result_vector_with_equals = equal_mask.select(
            result_vector, equal_vector
        )

        var nb_of_elements_to_store = _get_number_of_bytes_to_store_from_number_of_bytes_to_load[
            simd_width
        ](
            nb_of_elements_to_load
        )
        store_simd(
            result.unsafe_ptr() + len(result),
            result_vector_with_equals,
            nb_of_elements_to_store,
        )
        result.size += nb_of_elements_to_store
        input_index += input_simd_width


# Utility functions


fn _ceildiv_u8(a: Bytes, b: __type_of(a)) -> __type_of(a):
    return (a + b - 1) / b


fn _repeat_until[width: Int](v: SIMD) -> SIMD[v.type, width]:
    constrained[width >= v.size, "width must be at least v.size"]()

    @parameter
    if width == v.size:
        return rebind[SIMD[v.type, width]](v)
    return _repeat_until[width](v.join(v))


fn _rshift_bits_in_u16[shift: Int](input: Bytes) -> __type_of(input):
    var u16 = bitcast[DType.uint16, input.size // 2](input)
    var res = bit.rotate_bits_right[shift](u16)
    return bitcast[DType.uint8, input.size](res)


fn load_simd[
    width: Int
](pointer: UnsafePointer[Byte], len: Int) -> Bytes[width]:
    var result = Bytes[width]()
    var buffer_ptr = UnsafePointer.address_of(result).bitcast[Byte]()
    memcpy(dest=buffer_ptr, src=pointer, count=len)
    return result


fn store_simd(ptr: UnsafePointer[Byte], owned v: Bytes, len: Int):
    var buffer_ptr = UnsafePointer.address_of(v).bitcast[Byte]()
    memcpy(dest=ptr, src=buffer_ptr, count=len)
