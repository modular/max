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

"""Implements the StringSlice type.

You can import these APIs from the `utils.string_slice` module. For example:

```mojo
from utils import StringSlice
```
"""

from bit import count_leading_zeros
from utils import Span
from collections.string import _isspace, _atol, _atof
from collections import List, Optional
from memory import memcmp, UnsafePointer
from sys import simdwidthof, bitwidthof

alias StaticString = StringSlice[StaticConstantLifetime]
"""An immutable static string slice."""


fn _unicode_codepoint_utf8_byte_length(c: Int) -> Int:
    debug_assert(
        0 <= c <= 0x10FFFF, "Value: ", c, " is not a valid Unicode code point"
    )
    alias sizes = SIMD[DType.int32, 4](0, 0b0111_1111, 0b0111_1111_1111, 0xFFFF)
    return int((sizes < c).cast[DType.uint8]().reduce_add())


fn _shift_unicode_to_utf8(ptr: UnsafePointer[UInt8], c: Int, num_bytes: Int):
    """Shift unicode to utf8 representation.

    ### Unicode (represented as UInt32 BE) to UTF-8 conversion:
    - 1: 00000000 00000000 00000000 0aaaaaaa -> 0aaaaaaa
        - a
    - 2: 00000000 00000000 00000aaa aabbbbbb -> 110aaaaa 10bbbbbb
        - (a >> 6)  | 0b11000000, b         | 0b10000000
    - 3: 00000000 00000000 aaaabbbb bbcccccc -> 1110aaaa 10bbbbbb 10cccccc
        - (a >> 12) | 0b11100000, (b >> 6)  | 0b10000000, c        | 0b10000000
    - 4: 00000000 000aaabb bbbbcccc ccdddddd -> 11110aaa 10bbbbbb 10cccccc
    10dddddd
        - (a >> 18) | 0b11110000, (b >> 12) | 0b10000000, (c >> 6) | 0b10000000,
        d | 0b10000000
    """
    if num_bytes == 1:
        ptr[0] = UInt8(c)
        return

    var shift = 6 * (num_bytes - 1)
    var mask = UInt8(0xFF) >> (num_bytes + 1)
    var num_bytes_marker = UInt8(0xFF) << (8 - num_bytes)
    ptr[0] = ((c >> shift) & mask) | num_bytes_marker
    for i in range(1, num_bytes):
        shift -= 6
        ptr[i] = ((c >> shift) & 0b0011_1111) | 0b1000_0000


fn _utf8_byte_type(b: SIMD[DType.uint8, _], /) -> __type_of(b):
    """UTF-8 byte type.

    Returns:
        The byte type.

    Notes:

        - 0 -> ASCII byte.
        - 1 -> continuation byte.
        - 2 -> start of 2 byte long sequence.
        - 3 -> start of 3 byte long sequence.
        - 4 -> start of 4 byte long sequence.
    """
    return count_leading_zeros(~(b & UInt8(0b1111_0000)))


fn _is_newline_start(
    ptr: UnsafePointer[UInt8], read_ahead: Int = 1
) -> (Bool, Int):
    """Returns if the first item in the pointer is the start of
    a newline sequence, and its length.
    """
    # TODO add line and paragraph separator as StringLiteral
    # once Unicode escape sequences are accepted
    alias ` ` = UInt8(ord(" "))
    var rn = "\r\n"
    var next_line = List[UInt8](0xC2, 0x85)
    """TODO: \\x85"""
    var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8)
    """TODO: \\u2028"""
    var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9)
    """TODO: \\u2029"""

    var val = _utf8_byte_type(ptr[0])
    if val == 0:
        if read_ahead > 1:
            if memcmp(ptr, rn.unsafe_ptr(), 2) == 0:
                return True, 2
            _ = rn
        return ptr[0] != ` ` and _isspace(ptr[0]), 1
    elif val == 2 and read_ahead > 1:
        var comp = memcmp(ptr, next_line.unsafe_ptr(), 2) == 0
        _ = next_line
        return comp, 2
    elif val == 3 and read_ahead > 2:
        var comp = (
            memcmp(ptr, unicode_line_sep.unsafe_ptr(), 3) == 0
            or memcmp(ptr, unicode_paragraph_sep.unsafe_ptr(), 3) == 0
        )
        _ = unicode_line_sep, unicode_paragraph_sep
        return comp, 3
    return False, 1


@value
struct _StringSliceIter[
    is_mutable: Bool, //,
    lifetime: Lifetime[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for StringSlice

    Parameters:
        is_mutable: Whether the slice is mutable.
        lifetime: The lifetime of the underlying string data.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var continuation_bytes: Int
    var ptr: UnsafePointer[UInt8]
    var length: Int

    fn __init__(
        inout self, *, unsafe_pointer: UnsafePointer[UInt8], length: Int
    ):
        self.index = 0 if forward else length
        self.ptr = unsafe_pointer
        self.length = length
        self.continuation_bytes = 0
        for i in range(length):
            if _utf8_byte_type(unsafe_pointer[i]) == 1:
                self.continuation_bytes += 1

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> StringSlice[lifetime]:
        @parameter
        if forward:
            var byte_len = 1
            if self.continuation_bytes > 0:
                var byte_type = _utf8_byte_type(self.ptr[self.index])
                if byte_type != 0:
                    byte_len = int(byte_type)
                    self.continuation_bytes -= byte_len - 1
            self.index += byte_len
            return StringSlice[lifetime](
                unsafe_from_utf8_ptr=self.ptr + (self.index - byte_len),
                len=byte_len,
            )
        else:
            var byte_len = 1
            if self.continuation_bytes > 0:
                var byte_type = _utf8_byte_type(self.ptr[self.index - 1])
                if byte_type != 0:
                    while byte_type == 1:
                        byte_len += 1
                        var b = self.ptr[self.index - byte_len]
                        byte_type = _utf8_byte_type(b)
                    self.continuation_bytes -= byte_len - 1
            self.index -= byte_len
            return StringSlice[lifetime](
                unsafe_from_utf8_ptr=self.ptr + self.index, len=byte_len
            )

    @always_inline
    fn __hasmore__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index - self.continuation_bytes
        else:
            return self.index - self.continuation_bytes


struct StringSlice[
    is_mutable: Bool, //,
    lifetime: Lifetime[is_mutable].type,
](Stringable, Sized, Formattable):
    """
    A non-owning view to encoded string data.

    TODO:
    The underlying string data is guaranteed to be encoded using UTF-8.

    Parameters:
        is_mutable: Whether the slice is mutable.
        lifetime: The lifetime of the underlying string data.
    """

    var _slice: Span[UInt8, lifetime]

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __init__(
        inout self: StringSlice[StaticConstantLifetime], lit: StringLiteral
    ):
        """Construct a new string slice from a string literal.

        Args:
            lit: The literal to construct this string slice from.
        """
        # Since a StringLiteral has static lifetime, it will outlive
        # whatever arbitrary `lifetime` the user has specified they need this
        # slice to live for.
        # SAFETY:
        #   StringLiteral is guaranteed to use UTF-8 encoding.
        # FIXME(MSTDL-160):
        #   Ensure StringLiteral _actually_ always uses UTF-8 encoding.
        # TODO(#933): use when llvm intrinsics can be used at compile time
        # debug_assert(
        #     _is_valid_utf8(literal.unsafe_ptr(), literal.byte_length()),
        #     "StringLiteral doesn't have valid UTF-8 encoding",
        # )
        self = StaticString(
            unsafe_from_utf8_ptr=lit.unsafe_ptr(), len=lit.byte_length()
        )

    @always_inline
    fn __init__(inout self, *, owned unsafe_from_utf8: Span[UInt8, lifetime]):
        """
        Construct a new StringSlice from a sequence of UTF-8 encoded bytes.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.

        Args:
            unsafe_from_utf8: A slice of bytes encoded in UTF-8.
        """

        self._slice = unsafe_from_utf8^

    fn __init__(inout self, *, unsafe_from_utf8_strref: StringRef):
        """
        Construct a new StringSlice from a StringRef pointing to UTF-8 encoded
        bytes.

        Safety:
            - `unsafe_from_utf8_strref` MUST point to data that is valid for
              `lifetime`.
            - `unsafe_from_utf8_strref` MUST be valid UTF-8 encoded data.

        Args:
            unsafe_from_utf8_strref: A StringRef of bytes encoded in UTF-8.
        """
        var strref = unsafe_from_utf8_strref

        var byte_slice = Span[UInt8, lifetime](
            unsafe_ptr=strref.unsafe_ptr(),
            len=len(strref),
        )

        self = Self(unsafe_from_utf8=byte_slice)

    @always_inline
    fn __init__(
        inout self,
        *,
        unsafe_from_utf8_ptr: UnsafePointer[UInt8],
        len: Int,
    ):
        """
        Construct a StringSlice from a pointer to a sequence of UTF-8 encoded
        bytes and a length.

        Safety:
            - `unsafe_from_utf8_ptr` MUST point to at least `len` bytes of valid
              UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` must point to data that is live for the
              duration of `lifetime`.

        Args:
            unsafe_from_utf8_ptr: A pointer to a sequence of bytes encoded in
              UTF-8.
            len: The number of bytes of encoded data.
        """
        var byte_slice = Span[UInt8, lifetime](
            unsafe_ptr=unsafe_from_utf8_ptr,
            len=len,
        )

        self._slice = byte_slice

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Gets this slice as a standard `String`.

        Returns:
            The string representation of the slice.
        """
        return String(str_slice=self)

    fn __len__(self) -> Int:
        """Nominally returns the _length in Unicode codepoints_ (not bytes!).

        Returns:
            The length in Unicode codepoints.
        """
        var unicode_length = self.byte_length()

        for i in range(unicode_length):
            if _utf8_byte_type(self._slice[i]) == 1:
                unicode_length -= 1

        return unicode_length

    fn format_to(self, inout writer: Formatter):
        """
        Formats this string slice to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        writer.write_str(str_slice=self)

    fn __bool__(self) -> Bool:
        """Check if a string slice is non-empty.

        Returns:
           True if a string slice is non-empty, False otherwise.
        """
        return len(self._slice) > 0

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the lifetime.
    @__unsafe_disable_nested_lifetime_exclusivity
    fn __eq__(self, rhs: StringSlice) -> Bool:
        """Verify if a string slice is equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string slices are equal in length and contain the same elements, False otherwise.
        """
        if not self and not rhs:
            return True
        if len(self) != len(rhs):
            return False
        # same pointer and length, so equal
        if self._slice.unsafe_ptr() == rhs._slice.unsafe_ptr():
            return True
        for i in range(len(self)):
            if self._slice[i] != rhs._slice.unsafe_ptr()[i]:
                return False
        return True

    @always_inline
    fn __eq__(self, rhs: String) -> Bool:
        """Verify if a string slice is equal to a string.

        Args:
            rhs: The string to compare against.

        Returns:
            True if the string slice is equal to the input string in length and contain the same bytes, False otherwise.
        """
        return self == rhs.as_string_slice()

    @always_inline
    fn __eq__(self, rhs: StringLiteral) -> Bool:
        """Verify if a string slice is equal to a literal.

        Args:
            rhs: The literal to compare against.

        Returns:
            True if the string slice is equal to the input literal in length and contain the same bytes, False otherwise.
        """
        return self == rhs.as_string_slice()

    @__unsafe_disable_nested_lifetime_exclusivity
    @always_inline
    fn __ne__(self, rhs: StringSlice) -> Bool:
        """Verify if span is not equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string slices are not equal in length or contents, False otherwise.
        """
        return not self == rhs

    @always_inline
    fn __ne__(self, rhs: String) -> Bool:
        """Verify if span is not equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string and slice are not equal in length or contents, False otherwise.
        """
        return not self == rhs

    @always_inline
    fn __ne__(self, rhs: StringLiteral) -> Bool:
        """Verify if span is not equal to a literal.

        Args:
            rhs: The string literal to compare against.

        Returns:
            True if the slice is not equal to the literal in length or contents, False otherwise.
        """
        return not self == rhs

    fn __iter__(self) -> _StringSliceIter[lifetime]:
        """Iterate over elements of the string slice, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return _StringSliceIter[lifetime](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    fn __reversed__(self) -> _StringSliceIter[lifetime, False]:
        """Iterate backwards over the string, returning immutable references.

        Returns:
            A reversed iterator of references to the string elements.
        """
        return _StringSliceIter[lifetime, forward=False](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    @always_inline
    fn __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.
        If the string cannot be parsed as an int, an error is raised.

        Returns:
            An integer value that represents the string, or otherwise raises.
        """
        return _atol(self._strref_dangerous())

    @always_inline
    fn __float__(self) raises -> Float64:
        """Parses the string as a float point number and returns that value. If
        the string cannot be parsed as a float, an error is raised.

        Returns:
            A float value that represents the string, or otherwise raises.
        """
        return _atof(self._strref_dangerous())

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn as_bytes_span(self) -> Span[UInt8, lifetime]:
        """Get the sequence of encoded bytes as a slice of the underlying string.

        Returns:
            A slice containing the underlying sequence of encoded bytes.
        """
        return self._slice

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Gets a pointer to the first element of this string slice.

        Returns:
            A pointer pointing at the first element of this string slice.
        """

        return self._slice.unsafe_ptr()

    @always_inline
    fn byte_length(self) -> Int:
        """Get the length of this string slice in bytes.

        Returns:
            The length of this string slice in bytes.
        """

        return len(self.as_bytes_span())

    fn _strref_dangerous(self) -> StringRef:
        """Returns an inner pointer to the string as a StringRef.

        Safety:
            This functionality is extremely dangerous because Mojo eagerly
            releases strings.  Using this requires the use of the
            _strref_keepalive() method to keep the underlying string alive long
            enough.
        """
        return StringRef(self.unsafe_ptr(), self.byte_length())

    fn _strref_keepalive(self):
        """A no-op that keeps `self` alive through the call.  This
        can be carefully used with `_strref_dangerous()` to wield inner pointers
        without the string getting deallocated early.
        """
        pass

    fn _from_start(self, start: Int) -> Self:
        """Gets the `StringSlice` pointing to the substring after the specified slice start position.

        If start is negative, it is interpreted as the number of characters
        from the end of the string to start at.

        Args:
            start: Starting index of the slice.

        Returns:
            A `StringSlice` borrowed from the current string containing the
            characters of the slice starting at start.
        """

        var self_len = self.byte_length()

        var abs_start: Int
        if start < 0:
            # Avoid out of bounds earlier than the start
            # len = 5, start = -3,  then abs_start == 2, i.e. a partial string
            # len = 5, start = -10, then abs_start == 0, i.e. the full string
            abs_start = max(self_len + start, 0)
        else:
            # Avoid out of bounds past the end
            # len = 5, start = 2,   then abs_start == 2, i.e. a partial string
            # len = 5, start = 8,   then abs_start == 5, i.e. an empty string
            abs_start = min(start, self_len)

        debug_assert(
            abs_start >= 0, "strref absolute start must be non-negative"
        )
        debug_assert(
            abs_start <= self_len,
            "strref absolute start must be less than source String len",
        )

        # TODO: We assumes the StringSlice only has ASCII.
        # When we support utf-8 slicing, we should drop self._slice[abs_start:]
        # and use something smarter.
        return StringSlice(unsafe_from_utf8=self._slice[abs_start:])

    @always_inline
    fn format[*Ts: _CurlyEntryFormattable](self, *args: *Ts) raises -> String:
        """Format a template with `*args`.

        Args:
            args: The substitution values.

        Parameters:
            Ts: The types of substitution values that implement `Representable`
                and `Stringable` (to be changed and made more flexible).

        Returns:
            The template with the given values substituted.

        Examples:

        ```mojo
        # Manual indexing:
        print("{0} {1} {0}".format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print("{} {}".format(True, "hello world")) # True hello world
        ```
        .
        """
        return _FormatCurlyEntry.format(self, args)

    fn find(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """
        if not substr:
            return 0

        if self.byte_length() < substr.byte_length() + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack_str = self._from_start(start)

        var loc = stringref._memmem(
            haystack_str.unsafe_ptr(),
            haystack_str.byte_length(),
            substr.unsafe_ptr(),
            substr.byte_length(),
        )

        if not loc:
            return -1

        return int(loc) - int(self.unsafe_ptr())

    fn isspace(self) -> Bool:
        """Determines whether every character in the given StringSlice is a
        python whitespace String. This corresponds to Python's
        [universal separators](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `" \\t\\n\\r\\f\\v\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if the whole StringSlice is made up of whitespace characters
                listed above, otherwise False.
        """

        if self.byte_length() == 0:
            return False

        # TODO add line and paragraph separator as stringliteral
        # once Unicode escape sequences are accepted
        var next_line = List[UInt8](0xC2, 0x85)
        """TODO: \\x85"""
        var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8)
        """TODO: \\u2028"""
        var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9)
        """TODO: \\u2029"""

        for s in self:
            var no_null_len = s.byte_length()
            var ptr = s.unsafe_ptr()
            if no_null_len == 1 and _isspace(ptr[0]):
                continue
            elif (
                no_null_len == 2 and memcmp(ptr, next_line.unsafe_ptr(), 2) == 0
            ):
                continue
            elif no_null_len == 3 and (
                memcmp(ptr, unicode_line_sep.unsafe_ptr(), 3) == 0
                or memcmp(ptr, unicode_paragraph_sep.unsafe_ptr(), 3) == 0
            ):
                continue
            else:
                return False
        _ = next_line, unicode_line_sep, unicode_paragraph_sep
        return True

    fn splitlines(self, keepends: Bool = False) -> List[String]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\t\\n\\r\\r\\n\\f\\v\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """
        var output = List[String]()
        var length = self.byte_length()
        var current_offset = 0
        var ptr = self.unsafe_ptr()

        while current_offset < length:
            var eol_location = length - current_offset
            var eol_length = 0
            var curr_ptr = ptr.offset(current_offset)

            for i in range(current_offset, length):
                var read_ahead = 3 if i < length - 2 else (
                    2 if i < length - 1 else 1
                )
                var res = _is_newline_start(ptr.offset(i), read_ahead)
                if res[0]:
                    eol_location = i - current_offset
                    eol_length = res[1]
                    break

            var str_len: Int
            var end_of_string = False
            if current_offset >= length:
                end_of_string = True
                str_len = 0
            elif keepends:
                str_len = eol_location + eol_length
            else:
                str_len = eol_location

            output.append(
                String(Self(unsafe_from_utf8_ptr=curr_ptr, len=str_len))
            )

            if end_of_string:
                break
            current_offset += eol_location + eol_length

        return output^


# ===----------------------------------------------------------------------===#
# Utils
# ===----------------------------------------------------------------------===#


trait _Stringlike(CollectionElement, CollectionElementNew):
    """Trait intended to be used only with `String`, `StringLiteral` and
    `StringSlice`."""

    fn byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this string in bytes.

        Notes:
            This does not include the trailing null terminator in the count.
        """
        ...

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Get raw pointer to the underlying data.

        Returns:
            The raw pointer to the data.
        """
        ...


# ===----------------------------------------------------------------------===#
# Format method structures
# ===----------------------------------------------------------------------===#


trait _CurlyEntryFormattable(Stringable, Representable):
    """This trait is used by the `format()` method to support format specifiers.
    Currently, it is a composition of both `Stringable` and `Representable`
    traits i.e. a type to be formatted must implement both. In the future this
    will be less constrained.
    """

    pass


@value
struct _FormatCurlyEntry(CollectionElement, CollectionElementNew):
    """The struct that handles string-like formatting by curly braces entries.
    This is internal for the types: `String`, `StringLiteral` and `StringSlice`.
    """

    var first_curly: Int
    """The index of an opening brace around a substitution field."""
    var last_curly: Int
    """The index of an closing brace around a substitution field."""
    # TODO: ord("a") conversion flag not supported yet
    var conversion_flag: UInt8
    """The type of conversion for the entry: {ord("s"), ord("r")}."""
    var format_spec: Optional[_FormatSpec]
    """The format specifier."""
    # TODO: ord("a") conversion flag not supported yet
    alias supported_conversion_flags = SIMD[DType.uint8, 2](ord("s"), ord("r"))
    """Currently supported conversion flags: `__str__` and `__repr__`."""
    alias _FieldVariantType = Variant[String, Int, NoneType, Bool]
    """Purpose of the `Variant` `Self.field`:

    - `Int` for manual indexing: (value field contains `0`).
    - `NoneType` for automatic indexing: (value field contains `None`).
    - `String` for **kwargs indexing: (value field contains `foo`).
    - `Bool` for escaped curlies: (value field contains False for `{` or True
        for `}`).
    """
    var field: Self._FieldVariantType
    """Store the substitution field. See `Self._FieldVariantType` docstrings for
    more details."""
    alias _args_t = VariadicPack[element_trait=_CurlyEntryFormattable, *_]
    """Args types that are formattable by curly entry."""

    fn __init__(inout self, *, other: Self):
        self.first_curly = other.first_curly
        self.last_curly = other.last_curly
        self.conversion_flag = other.conversion_flag
        self.field = Self._FieldVariantType(other=other.field)
        self.format_spec = other.format_spec

    fn __init__(
        inout self,
        first_curly: Int,
        last_curly: Int,
        field: Self._FieldVariantType,
        conversion_flag: UInt8 = 0,
        format_spec: Optional[_FormatSpec] = None,
    ):
        self.first_curly = first_curly
        self.last_curly = last_curly
        self.field = field
        self.conversion_flag = conversion_flag
        self.format_spec = format_spec

    @always_inline
    fn is_escaped_brace(ref [_]self) -> Bool:
        return self.field.isa[Bool]()

    @always_inline
    fn is_kwargs_field(ref [_]self) -> Bool:
        return self.field.isa[String]()

    @always_inline
    fn is_automatic_indexing(ref [_]self) -> Bool:
        return self.field.isa[NoneType]()

    @always_inline
    fn is_manual_indexing(ref [_]self) -> Bool:
        return self.field.isa[Int]()

    @staticmethod
    fn format[T: _Stringlike](fmt_src: T, args: Self._args_t) raises -> String:
        alias len_pos_args = __type_of(args).__len__()
        entries, size_estimation = Self._create_entries(fmt_src, len_pos_args)
        var fmt_len = fmt_src.byte_length()
        var buf = String._buffer_type(capacity=fmt_len + size_estimation)
        buf.size = 1
        buf.unsafe_set(0, 0)
        var res = String(buf^)
        var offset = 0
        var ptr = fmt_src.unsafe_ptr()
        alias S = StringSlice[ImmutableAnyLifetime]

        @always_inline("nodebug")
        fn _build_slice(p: UnsafePointer[UInt8], start: Int, end: Int) -> S:
            return S(unsafe_from_utf8_ptr=p + start, len=end - start)

        var auto_arg_index = 0
        for e in entries:
            debug_assert(offset < fmt_len, "offset >= self.byte_length()")
            res += _build_slice(ptr, offset, e[].first_curly)
            Self._format_entry[len_pos_args](res, e[], auto_arg_index, args)
            offset = e[].last_curly + 1

        if offset < fmt_len:
            res += _build_slice(ptr, offset, fmt_len)

        return res^

    @staticmethod
    fn _create_entries[
        T: _Stringlike
    ](fmt_src: T, len_pos_args: Int) raises -> (List[Self], Int):
        """Returns a list of entries and its total estimated entry byte width.
        """
        var manual_indexing_count = 0
        var automatic_indexing_count = 0
        var raised_manual_index = Optional[Int](None)
        var raised_automatic_index = Optional[Int](None)
        var raised_kwarg_field = Optional[String](None)
        alias `}` = UInt8(ord("}"))
        alias `{` = UInt8(ord("{"))
        alias l_err = "there is a single curly { left unclosed or unescaped"
        alias r_err = "there is a single curly } left unclosed or unescaped"

        var entries = List[Self]()
        var start = Optional[Int](None)
        var skip_next = False
        var fmt_ptr = fmt_src.unsafe_ptr()
        var fmt_len = fmt_src.byte_length()
        var total_estimated_entry_byte_width = 0

        for i in range(fmt_len):
            if skip_next:
                skip_next = False
                continue
            if fmt_ptr[i] == `{`:
                if not start:
                    start = i
                    continue
                if i - start.value() != 1:
                    raise Error(l_err)
                # python escapes double curlies
                entries.append(Self(start.value(), i, field=False))
                start = None
                continue
            elif fmt_ptr[i] == `}`:
                if not start and (i + 1) < fmt_len:
                    # python escapes double curlies
                    if fmt_ptr[i + 1] == `}`:
                        entries.append(Self(i, i + 1, field=True))
                        total_estimated_entry_byte_width += 2
                        skip_next = True
                        continue
                elif not start:  # if it is not an escaped one, it is an error
                    raise Error(r_err)

                var start_value = start.value()
                var current_entry = Self(start_value, i, field=NoneType())

                if i - start_value != 1:
                    if current_entry._handle_field_and_break(
                        fmt_src,
                        len_pos_args,
                        i,
                        start_value,
                        automatic_indexing_count,
                        raised_automatic_index,
                        manual_indexing_count,
                        raised_manual_index,
                        raised_kwarg_field,
                        total_estimated_entry_byte_width,
                    ):
                        break
                else:  # automatic indexing
                    if automatic_indexing_count >= len_pos_args:
                        raised_automatic_index = automatic_indexing_count
                        break
                    automatic_indexing_count += 1
                    total_estimated_entry_byte_width += 8  # guessing
                entries.append(current_entry^)
                start = None

        if raised_automatic_index:
            raise Error("Automatic indexing require more args in *args")
        elif raised_kwarg_field:
            var val = raised_kwarg_field.value()
            raise Error("Index " + val + " not in kwargs")
        elif manual_indexing_count and automatic_indexing_count:
            raise Error("Cannot both use manual and automatic indexing")
        elif raised_manual_index:
            var val = str(raised_manual_index.value())
            raise Error("Index " + val + " not in *args")
        elif start:
            raise Error(l_err)
        return entries^, total_estimated_entry_byte_width

    fn _handle_field_and_break[
        T: _Stringlike
    ](
        inout self,
        fmt_src: T,
        len_pos_args: Int,
        i: Int,
        start_value: Int,
        inout automatic_indexing_count: Int,
        inout raised_automatic_index: Optional[Int],
        inout manual_indexing_count: Int,
        inout raised_manual_index: Optional[Int],
        inout raised_kwarg_field: Optional[String],
        inout total_estimated_entry_byte_width: Int,
    ) raises -> Bool:
        alias S = StringSlice[ImmutableAnyLifetime]

        @always_inline("nodebug")
        fn _build_slice(p: UnsafePointer[UInt8], start: Int, end: Int) -> S:
            return S(unsafe_from_utf8_ptr=p + start, len=end - start)

        var field = _build_slice(fmt_src.unsafe_ptr(), start_value + 1, i)
        var field_ptr = field.unsafe_ptr()
        var field_len = i - (start_value + 1)
        var exclamation_index = -1
        var idx = 0
        while idx < field_len:
            if field_ptr[idx] == ord("!"):
                exclamation_index = idx
                break
            idx += 1
        var new_idx = exclamation_index + 1
        if exclamation_index != -1:
            if new_idx == field_len:
                raise Error("Empty conversion flag.")
            var conversion_flag = field_ptr[new_idx]
            if field_len - new_idx > 1 or (
                conversion_flag not in Self.supported_conversion_flags
            ):
                var f = String(_build_slice(field_ptr, new_idx, field_len))
                _ = field^
                raise Error('Conversion flag "' + f + '" not recognised.')
            self.conversion_flag = conversion_flag
            field = _build_slice(field_ptr, 0, exclamation_index)
        else:
            new_idx += 1

        var extra = int(new_idx < field_len)
        var fmt_field = _build_slice(field_ptr, new_idx + extra, field_len)
        self.format_spec = _FormatSpec.parse(fmt_field)
        var w = int(self.format_spec.value().width) if self.format_spec else 0
        # fully guessing the byte width here to be at least 8 bytes per entry
        # minus the length of the whole format specification
        total_estimated_entry_byte_width += 8 * int(w > 0) + w - (field_len + 2)

        if field.byte_length() == 0:
            # an empty field, so it's automatic indexing
            if automatic_indexing_count >= len_pos_args:
                raised_automatic_index = automatic_indexing_count
                return True
            automatic_indexing_count += 1
        else:
            try:
                # field is a number for manual indexing:
                var number = int(field)
                self.field = number
                if number >= len_pos_args or number < 0:
                    raised_manual_index = number
                    return True
                manual_indexing_count += 1
            except e:
                alias unexp = "Not the expected error from atol"
                debug_assert("not convertible to integer" in str(e), unexp)
                # field is a keyword for **kwargs:
                var f = str(field)
                self.field = f
                raised_kwarg_field = f
                return True
        return False

    @staticmethod
    fn _format_entry[
        len_pos_args: Int
    ](
        inout res: String, e: Self, inout auto_idx: Int, args: Self._args_t
    ) raises:
        # TODO(#3403 and/or #3252): this function should be able to use
        # Formatter syntax when the type implements it, since it will give great
        # performance benefits. This also needs to be able to check if the given
        # args[i] conforms to the trait needed by the conversion_flag to avoid
        # needing to constraint that every type needs to conform to every trait.
        alias `r` = UInt8(ord("r"))
        alias `s` = UInt8(ord("s"))
        # alias `a` = UInt8(ord("a")) # TODO

        @parameter
        fn _format(idx: Int) raises:
            @parameter
            for i in range(len_pos_args):
                if i == idx:
                    var type_impls_repr = True  # TODO
                    var type_impls_str = True  # TODO
                    var type_impls_formatter_repr = True  # TODO
                    var type_impls_formatter_str = True  # TODO
                    var flag = e.conversion_flag
                    var empty = flag == 0 and not e.format_spec

                    var data: String
                    if empty and type_impls_formatter_str:
                        data = str(args[i])  # TODO: use formatter and return
                    elif empty and type_impls_str:
                        data = str(args[i])
                    elif flag == `s` and type_impls_formatter_str:
                        if empty:
                            # TODO: use formatter and return
                            pass
                        data = str(args[i])
                    elif flag == `s` and type_impls_str:
                        data = str(args[i])
                    elif flag == `r` and type_impls_formatter_repr:
                        if empty:
                            # TODO: use formatter and return
                            pass
                        data = repr(args[i])
                    elif flag == `r` and type_impls_repr:
                        data = repr(args[i])
                    elif e.format_spec:
                        e.format_spec.value().stringify(res, args[i])
                        return
                    else:
                        alias argnum = "Argument number: "
                        alias does_not = " does not implement the trait "
                        alias needed = "needed for conversion_flag: "
                        var flg = String(List[UInt8](flag, 0))
                        raise Error(argnum + str(i) + does_not + needed + flg)

                    if e.format_spec:
                        e.format_spec.value().format_string(res, data)
                    else:
                        res += data

        if e.is_escaped_brace():
            res += "}" if e.field[Bool] else "{"
        elif e.is_manual_indexing():
            _format(e.field[Int])
        elif e.is_automatic_indexing():
            _format(auto_idx)
            auto_idx += 1


@value
@register_passable("trivial")
struct _FormatSpec:
    """Store every field of the format specifier in a byte (e.g., ord("+") for
    sign). It is stored in a byte because every [format specifier](\
    https://docs.python.org/3/library/string.html#formatspec) is an ASCII
    character.
    """

    var fill: UInt8
    """If a valid align value is specified, it can be preceded by a fill
    character that can be any character and defaults to a space if omitted.
    """
    var align: UInt8
    """The meaning of the various alignment options is as follows:

    | Option | Meaning|
    |:-------|:-------|
    |'<' | Forces the field to be left-aligned within the available space
    (this is the default for most objects).|
    |'>' | Forces the field to be right-aligned within the available space
    (this is the default for numbers).|
    |'=' | Forces the padding to be placed after the sign (if any) but before
    the digits. This is used for printing fields in the form `+000000120`. This
    alignment option is only valid for numeric types. It becomes the default for
    numbers when `0` immediately precedes the field width.|
    |'^' | Forces the field to be centered within the available space.|
    """
    var sign: UInt8
    """The sign option is only valid for number types, and can be one of the
    following:

    | Option | Meaning|
    |:-------|:-------|
    |'+' | indicates that a sign should be used for both positive as well as
    negative numbers.|
    |'-' | indicates that a sign should be used only for negative numbers (this
    is the default behavior).|
    |space | indicates that a leading space should be used on positive numbers,
    and a minus sign on negative numbers.|
    """
    var coerce_z: Bool
    """The 'z' option coerces negative zero floating-point values to positive
    zero after rounding to the format precision. This option is only valid for
    floating-point presentation types.
    """
    var alternate_form: Bool
    """The alternate form is defined differently for different types. This
    option is only valid for types that implement the trait `# TODO: define
    trait`. For integers, when binary, octal, or hexadecimal output is used,
    this option adds the respective prefix '0b', '0o', '0x', or '0X' to the
    output value. For float and complex the alternate form causes the result of
    the conversion to always contain a decimal-point character, even if no
    digits follow it.
    """
    var width: UInt8
    """A decimal integer defining the minimum total field width, including any
    prefixes, separators, and other formatting characters. If not specified,
    then the field width will be determined by the content. When no explicit
    alignment is given, preceding the width field by a zero ('0') character
    enables sign-aware zero-padding for numeric types. This is equivalent to a
    fill character of '0' with an alignment type of '='.
    """
    var grouping_option: UInt8
    """The ',' option signals the use of a comma for a thousands separator. For
    a locale aware separator, use the 'n' integer presentation type instead. The
    '_' option signals the use of an underscore for a thousands separator for
    floating-point presentation types and for integer presentation type 'd'. For
    integer presentation types 'b', 'o', 'x', and 'X', underscores will be
    inserted every 4 digits. For other presentation types, specifying this
    option is an error.
    """
    var precision: UInt8
    """The precision is a decimal integer indicating how many digits should be
    displayed after the decimal point for presentation types 'f' and 'F', or
    before and after the decimal point for presentation types 'g' or 'G'. For
    string presentation types the field indicates the maximum field size - in
    other words, how many characters will be used from the field content. The
    precision is not allowed for integer presentation types.
    """
    var type: UInt8
    """Determines how the data should be presented. 
    
    The available integer presentation types are:

    | Option | Meaning|
    |:-------|:-------|
    |'b' |Binary format. Outputs the number in base 2.|
    |'c' |Character. Converts the integer to the corresponding unicode character
    before printing.|
    |'d' |Decimal Integer. Outputs the number in base 10.|
    |'o' |Octal format. Outputs the number in base 8.|
    |'x' |Hex format. Outputs the number in base 16, using lower-case letters
    for the digits above 9.|
    |'X' |Hex format. Outputs the number in base 16, using upper-case letters
    for the digits above 9. In case '#' is specified, the prefix '0x' will be
    upper-cased to '0X' as well.|
    |'n' |Number. This is the same as 'd', except that it uses the current
    locale setting to insert the appropriate number separator characters.|
    |None | The same as 'd'.|

    In addition to the above presentation types, integers can be formatted with
    the floating-point presentation types listed below (except 'n' and None).
    When doing so, float() is used to convert the integer to a floating-point
    number before formatting. 

    The available presentation types for float and Decimal values are:

    | Option | Meaning|
    |:-------|:-------|
    |'e' |Scientific notation. For a given precision p, formats the number in
    scientific notation with the letter `e` separating the coefficient from the
    exponent. The coefficient has one digit before and p digits after the
    decimal point, for a total of p + 1 significant digits. With no precision
    given, uses a precision of 6 digits after the decimal point for float, and
    shows all coefficient digits for Decimal. If no digits follow the decimal
    point, the decimal point is also removed unless the # option is used.|
    |'E' |Scientific notation. Same as 'e' except it uses an upper case `E` as
    the separator character.|
    |'f' |Fixed-point notation. For a given precision p, formats the number as a
    decimal number with exactly p digits following the decimal point. With no
    precision given, uses a precision of 6 digits after the decimal point for
    float, and uses a precision large enough to show all coefficient digits for
    Decimal. If no digits follow the decimal point, the decimal point is also
    removed unless the # option is used.|
    |'F' |Fixed-point notation. Same as 'f', but converts nan to NAN and inf to
    INF.|
    |'g' |General format. For a given precision p >= 1, this rounds the number
    to p significant digits and then formats the result in either fixed-point
    format or in scientific notation, depending on its magnitude. A precision of
    0 is treated as equivalent to a precision of 1.
    The precise rules are as follows: suppose that the result formatted with
    presentation type 'e' and precision p-1 would have exponent exp. Then, if
    m <= exp < p, where m is -4 for floats and -6 for Decimals, the number is
    formatted with presentation type 'f' and precision p-1-exp. Otherwise, the
    number is formatted with presentation type 'e' and precision p-1. In both
    cases insignificant trailing zeros are removed from the significand, and the
    decimal point is also removed if there are no remaining digits following it,
    unless the '#' option is used.
    With no precision given, uses a precision of 6 significant digits for float.
    For Decimal, the coefficient of the result is formed from the coefficient
    digits of the value; scientific notation is used for values smaller than
    1e-6 in absolute value and values where the place value of the least
    significant digit is larger than 1, and fixed-point notation is used
    otherwise.
    Positive and negative infinity, positive and negative zero, and nans, are
    formatted as inf, -inf, 0, -0 and nan respectively, regardless of the
    precision.|
    |'G' |General format. Same as 'g' except switches to 'E' if the number gets
    too large. The representations of infinity and NaN are uppercased, too.|
    |'n' |Number. This is the same as 'g', except that it uses the current
    locale setting to insert the appropriate number separator characters.|
    |'%' |Percentage. Multiplies the number by 100 and displays in fixed ('f')
    format, followed by a percent sign.|
    |None |For float this is like the 'g' type, except that when fixed-point
    notation is used to format the result, it always includes at least one digit
    past the decimal point, and switches to the scientific notation when
    exp >= p - 1. When the precision is not specified, the latter will be as
    large as needed to represent the given value faithfully.
    For Decimal, this is the same as either 'g' or 'G' depending on the value of
    context.capitals for the current decimal context.
    The overall effect is to match the output of str() as altered by the other
    format modifiers.|
    """

    fn __init__(
        inout self,
        fill: UInt8 = ord(" "),
        align: UInt8 = 0,
        sign: UInt8 = ord("-"),
        coerce_z: Bool = False,
        alternate_form: Bool = False,
        width: UInt8 = 0,
        grouping_option: UInt8 = 0,
        precision: UInt8 = 0,
        type: UInt8 = 0,
    ):
        """Construct a FormatSpec instance.

        Args:
            fill: Defaults to space.
            align: Defaults to 0 which is adjusted to the default for the arg
                type.
            sign: Defaults to `-`.
            coerce_z: Defaults to False.
            alternate_form: Defaults to False.
            width: Defaults to 0 which is adjusted to the default for the arg
                type.
            grouping_option: Defaults to 0 which is adjusted to the default for
                the arg type.
            precision: Defaults to 0 which is adjusted to the default for the
                arg type.
            type: Defaults to 0 which is adjusted to the default for the arg
                type.
        """
        self.fill = fill
        self.align = align
        self.sign = sign
        self.coerce_z = coerce_z
        self.alternate_form = alternate_form
        self.width = width
        self.grouping_option = grouping_option
        self.precision = precision
        self.type = type

    @staticmethod
    fn parse(fmt_str: StringSlice) -> Optional[Self]:
        """Parses the format spec string.

        Args:
            fmt_str: The StringSlice with the format spec.

        Returns:
            An instance of FormatSpec.
        """
        var f_len = fmt_str.byte_length()
        var f_ptr = fmt_str.unsafe_ptr()
        var colon_idx = -1
        var idx = 0
        while idx < f_len:
            if f_ptr[idx] == ord(":"):
                exclamation_index = idx
                break
            idx += 1

        if colon_idx == -1:
            return None

        # TODO: Future implementation of format specifiers
        return None

    fn stringify[
        T: _CurlyEntryFormattable
    ](self, inout res: String, item: T) raises:
        """Stringify a type according to its format specification.

        Args:
            res: The resulting String.
            item: The item to stringify.
        """
        var type_implements_float = True  # TODO
        var type_implements_float_raising = True  # TODO
        var type_implements_int = True  # TODO
        var type_implements_int_raising = True  # TODO

        # TODO: transform to int/float depending on format spec and stringify
        # with hex/bin/oct etc.
        res += str(item)

    fn format_string(self, inout res: String, item: String) raises:
        """Transform a String according to its format specification.

        Args:
            res: The resulting String.
            item: The item to format.
        """

        # TODO: align, fill, etc.
        res += item
