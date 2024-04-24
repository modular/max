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
"""Defines the List type.

You can import these APIs from the `collections` package. For example:

```mojo
from collections import List
```
"""


from builtin.value import StringableCollectionElement
from memory import UnsafePointer, Reference
from memory.unsafe_pointer import move_pointee, move_from_pointee


# ===----------------------------------------------------------------------===#
# Utilties
# ===----------------------------------------------------------------------===#


@always_inline
fn _max(a: Int, b: Int) -> Int:
    return a if a > b else b


# ===----------------------------------------------------------------------===#
# List
# ===----------------------------------------------------------------------===#


@value
struct _ListIter[
    T: CollectionElement,
    list_mutability: __mlir_type.`i1`,
    list_lifetime: AnyLifetime[list_mutability].type,
    forward: Bool = True,
]:
    """Iterator for List.

    Parameters:
        T: The type of the elements in the list.
        list_mutability: Whether the reference to the list is mutable.
        list_lifetime: The lifetime of the List
        forward: The iteration direction. `False` is backwards.
    """

    alias list_type = List[T]

    var index: Int
    var src: Reference[Self.list_type, list_mutability, list_lifetime]

    fn __iter__(self) -> Self:
        return self

    fn __next__(
        inout self,
    ) -> Reference[T, list_mutability, list_lifetime]:
        @parameter
        if forward:
            self.index += 1
            return self.src[].__get_ref[list_mutability, list_lifetime](
                self.index - 1
            )
        else:
            self.index -= 1
            return self.src[].__get_ref[list_mutability, list_lifetime](
                self.index
            )

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index


struct List[T: CollectionElement](CollectionElement, Sized, Boolable):
    """The `List` type is a dynamically-allocated list.

    It supports pushing and popping from the back resizing the underlying
    storage as needed.  When it is deallocated, it frees its memory.

    Parameters:
        T: The type of the elements.
    """

    var data: UnsafePointer[T]
    """The underlying storage for the list."""
    var size: Int
    """The number of elements in the list."""
    var capacity: Int
    """The amount of elements that can fit in the list without resizing it."""

    fn __init__(inout self):
        """Constructs an empty list."""
        self.data = UnsafePointer[T]()
        self.size = 0
        self.capacity = 0

    fn __init__(inout self, existing: Self):
        """Creates a deep copy of the given list.

        Args:
            existing: The list to copy.
        """
        self.__init__(capacity=existing.capacity)
        for e in existing:
            self.append(e[])

    fn __init__(inout self, *, capacity: Int):
        """Constructs a list with the given capacity.

        Args:
            capacity: The requested capacity of the list.
        """
        self.data = UnsafePointer[T].alloc(capacity)
        self.size = 0
        self.capacity = capacity

    # TODO: Avoid copying elements in once owned varargs
    # allow transfers.
    fn __init__(inout self, *values: T):
        """Constructs a list from the given values.

        Args:
            values: The values to populate the list with.
        """
        self = Self(capacity=len(values))
        for value in values:
            self.append(value[])

    fn __init__(
        inout self: Self, data: UnsafePointer[T], *, size: Int, capacity: Int
    ):
        """Constructs a list from a pointer, its size, and its capacity.

        Args:
            data: The pointer to the data.
            size: The number of elements in the list.
            capacity: The capacity of the list.
        """
        self.data = data
        self.size = size
        self.capacity = capacity

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing list into a new one.

        Args:
            existing: The existing list.
        """
        self.data = existing.data
        self.size = existing.size
        self.capacity = existing.capacity

    fn __copyinit__(inout self, existing: Self):
        """Creates a deepcopy of the given list.

        Args:
            existing: The list to copy.
        """
        self = Self(capacity=existing.capacity)
        for i in range(len(existing)):
            self.append(existing[i])

    @always_inline
    fn __del__(owned self):
        """Destroy all elements in the list and free its memory."""
        for i in range(self.size):
            destroy_pointee(self.data + i)
        if self.data:
            self.data.free()

    fn __len__(self) -> Int:
        """Gets the number of elements in the list.

        Returns:
            The number of elements in the list.
        """
        return self.size

    fn __bool__(self) -> Bool:
        """Checks whether the list has any elements or not.

        Returns:
            `False` if the list is empty, `True` if there is at least one element.
        """
        return len(self) > 0

    @always_inline
    fn _realloc(inout self, new_capacity: Int):
        var new_data = UnsafePointer[T].alloc(new_capacity)

        for i in range(self.size):
            move_pointee(src=self.data + i, dst=new_data + i)

        if self.data:
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    @always_inline
    fn append(inout self, owned value: T):
        """Appends a value to this list.

        Args:
            value: The value to append.
        """
        if self.size >= self.capacity:
            self._realloc(_max(1, self.capacity * 2))
        initialize_pointee_move(self.data + self.size, value^)
        self.size += 1

    @always_inline
    fn insert(inout self, i: Int, owned value: T):
        """Inserts a value to the list at the given index.
        `a.insert(len(a), value)` is equivalent to `a.append(value)`.

        Args:
            i: The index for the value.
            value: The value to insert.
        """
        debug_assert(i <= self.size, "insert index out of range")

        var normalized_idx = i
        if i < 0:
            normalized_idx = _max(0, len(self) + i)

        var earlier_idx = len(self)
        var later_idx = len(self) - 1
        self.append(value^)

        for _ in range(normalized_idx, len(self) - 1):
            var earlier_ptr = self.data + earlier_idx
            var later_ptr = self.data + later_idx

            var tmp = move_from_pointee(earlier_ptr)
            move_pointee(src=later_ptr, dst=earlier_ptr)
            initialize_pointee_move(later_ptr, tmp^)

            earlier_idx -= 1
            later_idx -= 1

    @always_inline
    fn extend(inout self, owned other: List[T]):
        """Extends this list by consuming the elements of `other`.

        Args:
            other: List whose elements will be added in order at the end of this list.
        """

        var final_size = len(self) + len(other)
        var other_original_size = len(other)

        self.reserve(final_size)

        # Defensively mark `other` as logically being empty, as we will be doing
        # consuming moves out of `other`, and so we want to avoid leaving `other`
        # in a partially valid state where some elements have been consumed
        # but are still part of the valid `size` of the list.
        #
        # That invalid intermediate state of `other` could potentially be
        # visible outside this function if a `__moveinit__()` constructor were
        # to throw (not currently possible AFAIK though) part way through the
        # logic below.
        other.size = 0

        var dest_ptr = self.data + len(self)

        for i in range(other_original_size):
            var src_ptr = other.data + i

            # This (TODO: optimistically) moves an element directly from the
            # `other` list into this list using a single `T.__moveinit()__`
            # call, without moving into an intermediate temporary value
            # (avoiding an extra redundant move constructor call).
            move_pointee(src=src_ptr, dst=dest_ptr)

            dest_ptr = dest_ptr + 1

        # Update the size now that all new elements have been moved into this
        # list.
        self.size = final_size

    @always_inline
    fn pop(inout self, i: Int = -1) -> T:
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.
        """
        debug_assert(-len(self) <= i < len(self), "pop index out of range")

        var normalized_idx = i
        if i < 0:
            normalized_idx += len(self)

        var ret_val = move_from_pointee(self.data + normalized_idx)
        for j in range(normalized_idx + 1, self.size):
            move_pointee(src=self.data + j, dst=self.data + j - 1)
        self.size -= 1
        if self.size * 4 < self.capacity:
            if self.capacity > 1:
                self._realloc(self.capacity // 2)
        return ret_val^

    @always_inline
    fn reserve(inout self, new_capacity: Int):
        """Reserves the requested capacity.

        If the current capacity is greater or equal, this is a no-op.
        Otherwise, the storage is reallocated and the date is moved.

        Args:
            new_capacity: The new capacity.
        """
        if self.capacity >= new_capacity:
            return
        self._realloc(new_capacity)

    @always_inline
    fn resize(inout self, new_size: Int, value: T):
        """Resizes the list to the given new size.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        list is appended with new values elements up to the requested size.

        Args:
            new_size: The new size.
            value: The value to use to populate new elements.
        """
        if new_size <= self.size:
            self.resize(new_size)
        else:
            self.reserve(new_size)
            for i in range(new_size, self.size):
                destroy_pointee(self.data + i)
            for i in range(self.size, new_size):
                initialize_pointee_copy(self.data + i, value)
            self.size = new_size

    @always_inline
    fn resize(inout self, new_size: Int):
        """Resizes the list to the given new size.

        With no new value provided, the new size must be smaller than or equal
        to the current one. Elements at the end are discarded.

        Args:
            new_size: The new size.
        """
        debug_assert(
            new_size <= self.size,
            (
                "New size must be smaller than or equal to current size when no"
                " new value is provided."
            ),
        )
        for i in range(new_size, self.size):
            destroy_pointee(self.data + i)
        self.size = new_size
        self.reserve(new_size)

    fn reverse(inout self):
        """Reverses the elements of the list."""

        self._reverse()

    # This method is private to avoid exposing the non-Pythonic `start` argument.
    @always_inline
    fn _reverse(inout self, start: Int = 0):
        """Reverses the elements of the list at positions after `start`.

        Args:
            start: A non-negative integer indicating the position after which to reverse elements.
        """

        # TODO(polish): Support a negative slice-like start position here that
        #               counts from the end.
        debug_assert(
            start >= 0,
            "List reverse start position must be non-negative",
        )

        var earlier_idx = start
        var later_idx = len(self) - 1

        var effective_len = len(self) - start
        var half_len = effective_len // 2

        for _ in range(half_len):
            var earlier_ptr = self.data + earlier_idx
            var later_ptr = self.data + later_idx

            var tmp = move_from_pointee(earlier_ptr)
            move_pointee(src=later_ptr, dst=earlier_ptr)
            initialize_pointee_move(later_ptr, tmp^)

            earlier_idx += 1
            later_idx -= 1

    fn clear(inout self):
        """Clears the elements in the list."""
        for i in range(self.size):
            destroy_pointee(self.data + i)
        self.size = 0

    fn steal_data(inout self) -> UnsafePointer[T]:
        """Take ownership of the underlying pointer from the list.

        Returns:
            The underlying data.
        """
        var ptr = self.data
        self.data = UnsafePointer[T]()
        self.size = 0
        self.capacity = 0
        return ptr

    fn __setitem__(inout self, i: Int, owned value: T):
        """Sets a list element at the given index.

        Args:
            i: The index of the element.
            value: The value to assign.
        """
        debug_assert(-self.size <= i < self.size, "index must be within bounds")

        var normalized_idx = i
        if i < 0:
            normalized_idx += len(self)

        destroy_pointee(self.data + normalized_idx)
        initialize_pointee_move(self.data + normalized_idx, value^)

    @always_inline
    fn _adjust_span(self, span: Slice) -> Slice:
        """Adjusts the span based on the list length."""
        var adjusted_span = span

        if adjusted_span.start < 0:
            adjusted_span.start = len(self) + adjusted_span.start

        if not adjusted_span._has_end():
            adjusted_span.end = len(self)
        elif adjusted_span.end < 0:
            adjusted_span.end = len(self) + adjusted_span.end

        if span.step < 0:
            var tmp = adjusted_span.end
            adjusted_span.end = adjusted_span.start - 1
            adjusted_span.start = tmp - 1

        return adjusted_span

    @always_inline
    fn __getitem__(self, span: Slice) -> Self:
        """Gets the sequence of elements at the specified positions.

        Args:
            span: A slice that specifies positions of the new list.

        Returns:
            A new list containing the list at the specified span.
        """

        var adjusted_span = self._adjust_span(span)
        var adjusted_span_len = len(adjusted_span)

        if not adjusted_span_len:
            return Self()

        var res = Self(capacity=len(adjusted_span))
        for i in range(len(adjusted_span)):
            res.append(self[adjusted_span[i]])

        return res^

    @always_inline
    fn __getitem__(self, i: Int) -> T:
        """Gets a copy of the list element at the given index.

        FIXME(lifetimes): This should return a reference, not a copy!

        Args:
            i: The index of the element.

        Returns:
            A copy of the element at the given index.
        """
        debug_assert(-self.size <= i < self.size, "index must be within bounds")

        var normalized_idx = i
        if i < 0:
            normalized_idx += len(self)

        return (self.data + normalized_idx)[]

    # TODO(30737): Replace __getitem__ with this as __refitem__, but lots of places use it
    fn __get_ref[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life]._mlir_type,
        i: Int,
    ) -> Reference[T, mutability, self_life]:
        """Gets a reference to the list element at the given index.

        Args:
            i: The index of the element.

        Returns:
            An immutable reference to the element at the given index.
        """
        var normalized_idx = i
        if i < 0:
            normalized_idx += Reference(self)[].size

        var offset_ptr = Reference(self)[].data + normalized_idx
        return offset_ptr[]

    fn __iter__[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life]._mlir_type,
    ) -> _ListIter[
        T, mutability, self_life
    ]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.
        """
        return _ListIter[T, mutability, self_life](0, Reference(self))

    fn __reversed__[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life]._mlir_type,
    ) -> _ListIter[
        T, mutability, self_life, False
    ]:
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.
        """
        var ref = Reference(self)
        return _ListIter[T, mutability, self_life, False](len(ref[]), ref)

    @staticmethod
    fn __str__[U: StringableCollectionElement](self: List[U]) -> String:
        """Returns a string representation of a `List`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_list = List[Int](1, 2, 3)
        print(__type_of(my_list).__str__(my_list))
        ```

        When the compiler supports conditional methods, then a simple `str(my_list)` will
        be enough.

        Args:
            self: The list to represent as a string.

        Parameters:
            U: The type of the elements in the list. Must implement the
              traits `Stringable` and `CollectionElement`.

        Returns:
            A string representation of the list.
        """
        # we do a rough estimation of the number of chars that we'll see
        # in the final string, we assume that str(x) will be at least one char.
        var minimum_capacity = (
            2  # '[' and ']'
            + len(self) * 3  # str(x) and ", "
            - 2  # remove the last ", "
        )
        var result = String(List[Int8](capacity=minimum_capacity))
        result += "["
        for i in range(len(self)):
            result += str(self[i])
            if i < len(self) - 1:
                result += ", "
        result += "]"
        return result
