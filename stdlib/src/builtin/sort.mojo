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
"""Implements the built-in `sort` function.

These are Mojo built-ins, so you don't need to import them.
"""

from bit import ctlz
from collections import List
from memory import Pointer, UnsafePointer
from sys import bitwidthof

# ===----------------------------------------------------------------------===#
# sort
# ===----------------------------------------------------------------------===#

alias _cmp_fn_type = fn[type: AnyRegType] (type, type) capturing -> Bool


fn _insertion_sort[
    type: AnyRegType, cmp_fn: _cmp_fn_type
](array: Pointer[type], start: Int, end: Int):
    """Sort the array[start:end] slice"""

    for i in range(start + 1, end):
        var value = array[i]
        var j = i

        # Find the placement of the value in the array, shifting as we try to
        # find the position. Throughout, we assume array[start:i] has already
        # been sorted.
        while j > start and not cmp_fn[type](array[j - 1], value):
            array[j] = array[j - 1]
            j -= 1

        array[j] = value


fn _insertion_sort[
    type: CollectionElement, cmp_fn: fn (type, type) capturing -> Bool
](array: UnsafePointer[type], start: Int, end: Int):
    """Sort the array[start:end] slice"""

    for i in range(start + 1, end):
        var value = array[i]
        var j = i

        # Find the placement of the value in the array, shifting as we try to
        # find the position. Throughout, we assume array[start:i] has already
        # been sorted.
        while j > start and not cmp_fn(array[j - 1], value):
            array[j] = array[j - 1]
            j -= 1

        array[j] = value


@always_inline
fn _partition[
    type: AnyRegType, cmp_fn: _cmp_fn_type
](array: Pointer[type], start: Int, end: Int) -> Int:
    if start == end:
        return end

    var pivot = start + (end - start) // 2

    var pivot_value = array[pivot]

    var left = start
    var right = end - 2

    swap(array[pivot], array[end - 1])

    while left < right:
        if cmp_fn[type](array[left], pivot_value):
            left += 1
        elif not cmp_fn[type](array[right], pivot_value):
            right -= 1
        else:
            swap(array[left], array[right])

    if cmp_fn[type](array[right], pivot_value):
        right += 1
    swap(array[end - 1], array[right])
    return right


@always_inline
fn _partition[
    type: CollectionElement, cmp_fn: fn (type, type) capturing -> Bool
](array: UnsafePointer[type], start: Int, end: Int) -> Int:
    if start == end:
        return end

    var pivot = start + (end - start) // 2

    var pivot_value = array[pivot]

    var left = start
    var right = end - 2

    swap(array[pivot], array[end - 1])

    while left < right:
        if cmp_fn(array[left], pivot_value):
            left += 1
        elif not cmp_fn(array[right], pivot_value):
            right -= 1
        else:
            swap(array[left], array[right])

    if cmp_fn(array[right], pivot_value):
        right += 1
    swap(array[end - 1], array[right])
    return right


fn _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    var log2 = int((bitwidthof[DType.index]() - 1) ^ ctlz(size | 1))
    return max(2, log2)


fn _quicksort[
    type: AnyRegType, cmp_fn: _cmp_fn_type
](array: Pointer[type], size: Int):
    if size == 0:
        return

    var stack = List[Int](capacity=_estimate_initial_height(size))
    stack.append(0)
    stack.append(size)
    while len(stack) > 0:
        var end = stack.pop()
        var start = stack.pop()

        var len = end - start
        if len < 2:
            continue

        if len == 2:
            _small_sort[2, type, cmp_fn](array + start)
            continue

        if len == 3:
            _small_sort[3, type, cmp_fn](array + start)
            continue

        if len == 4:
            _small_sort[4, type, cmp_fn](array + start)
            continue

        if len == 5:
            _small_sort[5, type, cmp_fn](array + start)
            continue

        if len < 32:
            _insertion_sort[type, cmp_fn](array, start, end)
            continue

        var pivot = _partition[type, cmp_fn](array, start, end)

        stack.append(pivot + 1)
        stack.append(end)

        stack.append(start)
        stack.append(pivot)


fn _quicksort[
    type: CollectionElement, cmp_fn: fn (type, type) capturing -> Bool
](array: UnsafePointer[type], size: Int):
    if size == 0:
        return

    var stack = List[Int](capacity=_estimate_initial_height(size))
    stack.append(0)
    stack.append(size)
    while len(stack) > 0:
        var end = stack.pop()
        var start = stack.pop()

        var len = end - start
        if len < 2:
            continue

        if len < 8:
            _insertion_sort[type, cmp_fn](array, start, end)
            continue

        var pivot = _partition[type, cmp_fn](array, start, end)

        stack.append(pivot + 1)
        stack.append(end)

        stack.append(start)
        stack.append(pivot)


# ===----------------------------------------------------------------------===#
# partition
# ===----------------------------------------------------------------------===#
fn partition[
    type: AnyRegType, cmp_fn: _cmp_fn_type
](buff: Pointer[type], k: Int, size: Int):
    """Partition the input vector inplace such that first k elements are the
    largest (or smallest if cmp_fn is <= operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        type: DType of the underlying data.
        cmp_fn: Comparison functor of type, type) capturing -> Bool type.

    Args:
        buff: Input buffer.
        k: Index of the partition element.
        size: The length of the buffer.
    """
    var stack = List[Int](capacity=_estimate_initial_height(size))
    stack.append(0)
    stack.append(size)
    while len(stack) > 0:
        var end = stack.pop()
        var start = stack.pop()
        var pivot = _partition[type, cmp_fn](buff, start, end)
        if pivot == k:
            break
        elif k < pivot:
            stack.append(start)
            stack.append(pivot)
        else:
            stack.append(pivot + 1)
            stack.append(end)


# ===----------------------------------------------------------------------===#
# sort
# ===----------------------------------------------------------------------===#


fn sort(inout buff: Pointer[Int], len: Int):
    """Sort the vector inplace.
    The function doesn't return anything, the vector is updated inplace.

    Args:
        buff: Input buffer.
        len: The length of the buffer.
    """

    @parameter
    fn _less_than_equal[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    _quicksort[Int, _less_than_equal](buff, len)


fn sort[type: DType](inout buff: Pointer[Scalar[type]], len: Int):
    """Sort the vector inplace.
    The function doesn't return anything, the vector is updated inplace.

    Parameters:
        type: DType of the underlying data.

    Args:
        buff: Input buffer.
        len: The length of the buffer.
    """

    @parameter
    fn _less_than_equal[ty: AnyRegType](lhs: ty, rhs: ty) -> Bool:
        return rebind[Scalar[type]](lhs) <= rebind[Scalar[type]](rhs)

    _quicksort[Scalar[type], _less_than_equal](buff, len)


fn sort(inout v: List[Int]):
    """Sort the vector inplace.
    The function doesn't return anything, the vector is updated inplace.

    Args:
        v: Input integer vector to sort.
    """
    # Downcast any pointer to register-passable pointer.
    var ptr = rebind[Pointer[Int]](v.data)
    sort(ptr, len(v))


fn sort[type: DType](inout v: List[Scalar[type]]):
    """Sort the vector inplace.
    The function doesn't return anything, the vector is updated inplace.

    Parameters:
        type: DType of the underlying data.

    Args:
        v: Input vector to sort.
    """

    var ptr = rebind[Pointer[Scalar[type]]](v.data)
    sort[type](ptr, len(v))


fn sort[
    type: CollectionElement,
    cmp_fn: fn (type, type) capturing -> Bool,
](inout v: List[type]):
    """Sort the vector inplace.
    The function doesn't return anything, the vector is updated inplace.

    Parameters:
        type: DType of the underlying data.
        cmp_fn: The comparison function.

    Args:
        v: Input vector to sort.
    """

    _quicksort[type, cmp_fn](v.data, len(v))


# ===----------------------------------------------------------------------===#
# sort networks
# ===----------------------------------------------------------------------===#


@always_inline
fn _sort2[
    type: AnyRegType, cmp_fn: _cmp_fn_type
](array: Pointer[type], offset0: Int, offset1: Int):
    var a = array[offset0]
    var b = array[offset1]
    if not cmp_fn[type](a, b):
        array[offset0] = b
        array[offset1] = a


@always_inline
fn _sort_partial_3[
    type: AnyRegType, cmp_fn: _cmp_fn_type
](array: Pointer[type], offset0: Int, offset1: Int, offset2: Int):
    var a = array[offset0]
    var b = array[offset1]
    var c = array[offset2]
    var r = cmp_fn[type](c, a)
    var t = c if r else a
    if r:
        array[offset2] = a
    if cmp_fn[type](b, t):
        array[offset0] = b
        array[offset1] = t
    elif r:
        array[offset0] = t


@always_inline
fn _small_sort[
    n: Int, type: AnyRegType, cmp_fn: _cmp_fn_type
](array: Pointer[type]):
    @parameter
    if n == 2:
        _sort2[type, cmp_fn](array, 0, 1)
        return

    @parameter
    if n == 3:
        _sort2[type, cmp_fn](array, 1, 2)
        _sort_partial_3[type, cmp_fn](array, 0, 1, 2)
        return

    @parameter
    if n == 4:
        _sort2[type, cmp_fn](array, 0, 2)
        _sort2[type, cmp_fn](array, 1, 3)
        _sort2[type, cmp_fn](array, 0, 1)
        _sort2[type, cmp_fn](array, 2, 3)
        _sort2[type, cmp_fn](array, 1, 2)
        return

    @parameter
    if n == 5:
        _sort2[type, cmp_fn](array, 0, 1)
        _sort2[type, cmp_fn](array, 3, 4)
        _sort_partial_3[type, cmp_fn](array, 2, 3, 4)
        _sort2[type, cmp_fn](array, 1, 4)
        _sort_partial_3[type, cmp_fn](array, 0, 2, 3)
        _sort_partial_3[type, cmp_fn](array, 1, 2, 3)
        return
