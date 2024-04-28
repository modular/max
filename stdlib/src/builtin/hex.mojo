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

"""Provides the `hex` function.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import List, Optional
from utils.inlined_string import _ArrayMem

alias _DEFAULT_DIGIT_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


@always_inline
fn hex[T: Intable](value: T) -> String:
    """Returns the hex string represention of the given integer.

    The hexadecimal representation is a base-16 encoding of the integer value.

    The returned string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Parameters:
        T: The intable type to represent in hexadecimal.

    Args:
        value: The integer value to format.

    Returns:
        A string containing the hex representation of the given integer.
    """

    try:
        return _format_int(int(value), 16, prefix="0x")
    except e:
        # This should not be reachable as _format_int only throws if we pass
        # incompatible radix and custom digit chars, which we aren't doing
        # above.
        return abort[String](
            "unexpected exception formatting value as hexadecimal: " + str(e)
        )


# ===----------------------------------------------------------------------===#
# Integer formatting utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _abs(x: SIMD) -> __type_of(x):
    return (x > 0).select(x, -x)


fn _format_int(
    value: Int64,
    radix: Int = 10,
    digit_chars: StringLiteral = _DEFAULT_DIGIT_CHARS,
    prefix: StringLiteral = "",
) raises -> String:
    var string = String()
    var fmt = string._unsafe_to_formatter()

    _write_int(fmt, value, radix, digit_chars, prefix)

    return string^


@always_inline
fn _write_int(
    inout fmt: Formatter,
    value: Int64,
    radix: Int = 10,
    digit_chars: StringLiteral = _DEFAULT_DIGIT_CHARS,
    prefix: StringLiteral = "",
) raises:
    var err = _try_write_int(fmt, value, radix, digit_chars, prefix)
    if err:
        raise err.value()[]


@always_inline
fn _try_write_int(
    inout fmt: Formatter,
    value: Int64,
    radix: Int = 10,
    digit_chars: StringLiteral = _DEFAULT_DIGIT_CHARS,
    prefix: StringLiteral = "",
) -> Optional[Error]:
    """Writes a formatted string representation of the given integer using the specified radix.

    The maximum supported radix is 36 unless a custom `digit_chars` mapping is
    provided.
    """

    #
    # Check that the radix and available digit characters are valid
    #

    if radix < 2:
        return Error("Unable to format integer to string with radix < 2")

    if radix > len(digit_chars):
        return Error(
            "Unable to format integer to string when provided radix is larger "
            "than length of available digit value characters"
        )

    if not len(digit_chars) >= 2:
        return Error(
            "Unable to format integer to string when provided digit_chars"
            " mapping len is not >= 2"
        )

    #
    # Process the integer value into its corresponding digits
    #

    # TODO(#26444, Unicode support): Get an array of Character, not bytes.
    var digit_chars_array = digit_chars.data()

    # Prefix a '-' if the original int was negative and make positive.
    if value < 0:
        fmt.write_str("-")

    # Add the custom number prefix, e.g. "0x" commonly used for hex numbers.
    # This comes *after* the minus sign, if present.
    fmt.write_str(prefix)

    if value == 0:
        var zero = StringRef(digit_chars_array, 1)
        fmt.write_str(zero)
        return

    #
    # Create a buffer to store the formatted value
    #

    # Stack allocate enough bytes to store any formatted 64-bit integer
    alias CAPACITY: Int = 64

    var buf = _ArrayMem[Int8, CAPACITY]()

    # Start the buf pointer at the end. We will write the least-significant
    # digits later in the buffer, and then decrement the pointer to move
    # earlier in the buffer as we write the more-significant digits.
    var offset = CAPACITY - 1

    #
    # Write the digits of the number
    #

    var remaining_int = value

    @parameter
    fn process_digits[get_digit_value: fn () capturing -> Int64]():
        while remaining_int:
            var digit_value = get_digit_value()

            # Write the char representing the value of the least significant
            # digit.
            buf[offset] = digit_chars_array[digit_value]

            # Position the offset to write the next digit.
            offset -= 1

            # Drop the least significant digit
            remaining_int /= radix

    if remaining_int >= 0:

        @parameter
        fn pos_digit_value() -> Int64:
            return remaining_int % radix

        process_digits[pos_digit_value]()
    else:

        @parameter
        fn neg_digit_value() -> Int64:
            return _abs(remaining_int % -radix)

        process_digits[neg_digit_value]()

    # Re-add +1 byte since the loop ended so we didn't write another char.
    offset += 1

    var buf_ptr = buf.as_ptr() + offset

    # Calculate the length of the buffer we've filled. This is the number of
    # bytes from our final `buf_ptr` to the end of the buffer.
    var len = CAPACITY - offset

    var strref = StringRef(rebind[Pointer[Int8]](buf_ptr), len)

    fmt.write_str(strref)

    return None
