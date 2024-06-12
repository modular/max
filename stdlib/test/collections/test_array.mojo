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
# RUN: %mojo %s

from testing import assert_equal, assert_false, assert_true, assert_raises

from collections.array import Array


fn test_array() raises:
    var array = Array[DType.int8, 5]()

    for i in range(5):
        array.append(i)

    # Verify it's iterable
    var index = 0
    for element in array:
        assert_equal(array[index], element)
        index += 1

    assert_equal(5, len(array))

    # Can assign a specified index in static data range via `setitem`
    array[2] = -2
    assert_equal(0, array[0])
    assert_equal(1, array[1])
    assert_equal(-2, array[2])
    assert_equal(3, array[3])
    assert_equal(4, array[4])

    assert_equal(0, array[-5])
    assert_equal(3, array[-2])
    assert_equal(4, array[-1])

    array[-5] = 5
    assert_equal(5, array[-5])
    array[-2] = 3
    assert_equal(3, array[-2])
    array[-1] = 7
    assert_equal(7, array[-1])

    # Can't assign past the static size
    for i in range(5, 10):
        array.append(i)

    assert_equal(5, len(array))
    assert_equal(9, array[-1])

    # clearing zeroes the array
    array.clear()
    assert_equal(0, len(array))
    for i in range(5):
        assert_equal(0, array[i])

    var arr = Array[DType.int8]()

    for i in range(5):
        arr.append(i)

    assert_equal(5, len(arr))
    assert_equal(0, arr[0])
    assert_equal(1, arr[1])
    assert_equal(2, arr[2])
    assert_equal(3, arr[3])
    assert_equal(4, arr[4])

    assert_equal(0, arr[-5])
    assert_equal(3, arr[-2])
    assert_equal(4, arr[-1])

    arr[2] = -2
    assert_equal(-2, arr[2])

    arr[-5] = 5
    assert_equal(5, arr[-5])
    arr[-2] = 3
    assert_equal(3, arr[-2])
    arr[-1] = 7
    assert_equal(7, arr[-1])


fn test_array_with_default() raises:
    var array = Array[DType.int8]()

    for i in range(5):
        array.append(i)

    assert_equal(5, len(array))

    array[2] = -2

    assert_equal(0, array[0])
    assert_equal(1, array[1])
    assert_equal(-2, array[2])
    assert_equal(3, array[3])
    assert_equal(4, array[4])

    for j in range(5, 10):
        array.append(j)

    assert_equal(10, len(array))

    assert_equal(5, array[5])

    array[5] = -2
    assert_equal(-2, array[5])

    array.clear()
    assert_equal(0, len(array))


fn test_indexing_vec() raises:
    var array = Array[DType.int8]()
    for i in range(5):
        array.append(i)
    assert_equal(0, array[0])
    assert_equal(1, array[True])
    assert_equal(2, array[2])


fn test_mojo_issue_698() raises:
    var arr = Array[DType.float64, 5]()
    for i in range(5):
        arr.append(i)

    assert_equal(0.0, arr[0])
    assert_equal(1.0, arr[1])
    assert_equal(2.0, arr[2])
    assert_equal(3.0, arr[3])
    assert_equal(4.0, arr[4])


fn test_array_to_bool_conversion() raises:
    assert_false(Array[DType.int8]())
    assert_true(Array[DType.int8](0))
    assert_true(Array[DType.int8](0, 1))
    assert_true(Array[DType.int8](1))


fn test_array_pop() raises:
    var arr = Array[DType.int8]()
    # Test pop with index
    for i in range(6):
        arr.append(i)

    # try poping from index 3 for 3 times
    for i in range(3, 6):
        assert_equal(i, arr.pop(3))

    # should have 3 elements now
    assert_equal(3, len(arr))
    assert_equal(0, arr[0])
    assert_equal(1, arr[1])
    assert_equal(2, arr[2])

    # Test pop with negative index
    for i in range(0, 2):
        assert_equal(i, arr.pop(-len(arr)))

    # test default index as well
    assert_equal(2, arr.pop())
    arr.append(2)
    assert_equal(2, arr.pop())

    # arr should be empty now
    assert_equal(0, len(arr))


fn test_array_variadic_constructor() raises:
    var l = Array[DType.int8](2, 4, 6)
    assert_equal(3, len(l))
    assert_equal(2, l[0])
    assert_equal(4, l[1])
    assert_equal(6, l[2])

    l.append(8)
    assert_equal(4, len(l))
    assert_equal(8, l[3])


fn test_array_insert() raises:
    #
    # Test the list [1, 2, 3] created with insert
    #

    var v1 = Array[DType.int8]()
    v1.insert(len(v1), 1)
    v1.insert(1, 2)
    v1.insert(len(v1), 3)

    assert_equal(len(v1), 3)
    assert_equal(v1[0], 1)
    assert_equal(v1[1], 2)
    assert_equal(v1[2], 3)

    #
    # Test the list [1, 2, 3, 4, 5] created with negative and positive index
    #

    var v2 = Array[DType.int8, 5]()
    v2.insert(-1729, 2)
    v2.insert(len(v2), 3)
    v2.insert(len(v2), 5)
    v2.insert(-5, 1)
    v2.insert(-2, 4)

    assert_equal(len(v2), 5)
    assert_equal(v2[0], 1)
    assert_equal(v2[1], 2)
    assert_equal(v2[2], 3)
    assert_equal(v2[3], 4)
    assert_equal(v2[4], 5)

    #
    # Test the list [1, 2, 3, 4] created with negative index
    #

    var v3 = Array[DType.int8, 4]()
    v3.insert(-11, 4)
    v3.insert(-13, 3)
    v3.insert(-17, 2)
    v3.insert(-19, 1)

    assert_equal(len(v3), 4)
    assert_equal(v3[0], 1)
    assert_equal(v3[1], 2)
    assert_equal(v3[2], 3)
    assert_equal(v3[3], 4)

    #
    # Test the list [1, 2, 3, 4, 5, 6, 7, 8] created with insert
    #

    var v4 = Array[DType.int8]()
    for i in range(4):
        v4.insert(0, 4 - i)
        v4.insert(len(v4), 4 + i + 1)

    for i in range(len(v4)):
        assert_equal(v4[i], i + 1)


fn test_array_index() raises:
    var test_array_a = Array[DType.int8](10, 20, 30, 40, 50)

    # Basic Functionality Tests
    assert_equal(test_array_a.index(10).value(), 0)
    assert_equal(test_array_a.index(30).value(), 2)
    assert_equal(test_array_a.index(50).value(), 4)
    assert_false(test_array_a.index(60))
    # Tests With Start Parameter
    assert_equal(test_array_a.index(30, start=1).value(), 2)
    assert_equal(test_array_a.index(30, start=-4).value(), 2)
    assert_equal(test_array_a.index(30, start=-1000).value(), 2)
    assert_false(test_array_a.index(30, start=3))
    assert_false(test_array_a.index(30, start=5))
    # Tests With Start and End Parameters
    assert_equal(test_array_a.index(30, start=1, stop=3).value(), 2)
    assert_equal(test_array_a.index(30, start=-4, stop=-2).value(), 2)
    assert_equal(test_array_a.index(30, start=-1000, stop=1000).value(), 2)
    assert_false(test_array_a.index(30, start=1, stop=2))
    assert_false(test_array_a.index(30, start=3, stop=1))
    # Tests With End Parameter Only
    assert_equal(test_array_a.index(30, stop=3).value(), 2)
    assert_equal(test_array_a.index(30, stop=-2).value(), 2)
    assert_equal(test_array_a.index(30, stop=1000).value(), 2)
    assert_false(test_array_a.index(30, stop=1))
    assert_false(test_array_a.index(30, stop=2))
    assert_false(test_array_a.index(60, stop=50))
    # Edge Cases and Special Conditions
    assert_equal(test_array_a.index(10, start=-5, stop=-1).value(), 0)
    assert_equal(test_array_a.index(10, start=0, stop=50).value(), 0)
    assert_equal(test_array_a.index(50, start=-5, stop=-1).value(), 4)
    assert_equal(test_array_a.index(50, start=0, stop=-1).value(), 4)
    assert_false(test_array_a.index(10, start=-4, stop=-1))
    assert_false(test_array_a.index(10, start=5, stop=50))
    assert_false(Array[DType.int8]().index(10))
    # print("5")
    # # Test empty slice
    assert_false(test_array_a.index(10, start=1, stop=1))
    # Test empty slice with 0 start and end
    assert_false(test_array_a.index(10, start=0, stop=0))
    var test_array_b = Array[DType.int8](10, 20, 30, 20, 10)

    # Test finding the first occurrence of an item
    assert_equal(test_array_b.index(10).value(), 0)
    assert_equal(test_array_b.index(20).value(), 1)
    # Test skipping the first occurrence with a start parameter
    assert_equal(test_array_b.index(20, start=2).value(), 3)
    # Test constraining search with start and end, excluding last occurrence
    assert_false(test_array_b.index(10, start=1, stop=4))
    # Test search within a range that includes multiple occurrences
    assert_equal(test_array_b.index(20, start=1, stop=4).value(), 1)
    # Verify error when constrained range excludes occurrences
    assert_false(test_array_b.index(20, start=4, stop=5))


fn test_array_extend() raises:
    #
    # Test extending the list [1, 2, 3] with itself
    #

    var vec = Array[DType.int8]()
    vec.append(1)
    vec.append(2)
    vec.append(3)

    assert_equal(len(vec), 3)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)

    var copy = vec
    vec.extend(copy)

    # vec == [1, 2, 3, 1, 2, 3]
    assert_equal(len(vec), 6)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)
    assert_equal(vec[3], 1)
    assert_equal(vec[4], 2)
    assert_equal(vec[5], 3)


fn test_array_iter() raises:
    var vs = Array[DType.index]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    # Borrow immutably
    fn sum(vs: Array[DType.index]) -> Int:
        var sum = 0
        for v in vs:
            sum += int(v)
        return sum

    assert_equal(6, sum(vs))


fn test_array_iter_not_mutable() raises:
    var vs = Array(1, 2, 3)

    # should not mutate
    for v in vs:
        v += 1
    var sum = 0
    for v in vs:
        sum += int(v)
    assert_equal(6, sum)


fn test_array_broadcast_ops() raises:
    alias arr = Array[DType.uint8, 3]
    var vs = arr(1, 2, 3)
    # should apply to all
    vs += 1
    assert_equal(9, vs.sum())
    vs -= 1
    assert_equal(6, vs.sum())
    vs *= 2
    assert_equal(12, vs.sum())
    assert_equal(0, (arr(2, 2, 2) % 2).sum())
    assert_equal(0, (arr(2, 2, 2) // 2).sum())
    assert_equal(4 * 3, (arr(2, 2, 2) ** 2).sum())


fn test_array_span() raises:
    var vs = Array[DType.int8, 3](1, 2, 3)

    var es = vs[1:]
    assert_equal(es[0], 2)
    assert_equal(es[1], 3)
    assert_equal(len(es), 2)

    es = vs[:-1]
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(len(es), 2)

    es = vs[1:-1:1]
    assert_equal(es[0], 2)
    assert_equal(len(es), 1)

    es = vs[::-1]
    assert_equal(es[0], 3)
    assert_equal(es[1], 2)
    assert_equal(es[2], 1)
    assert_equal(len(es), 3)

    es = vs[:]
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(es[2], 3)
    assert_equal(len(es), 3)


fn test_array_boolable() raises:
    assert_true(Array[DType.int8](1))
    assert_false(Array[DType.int8]())


fn test_constructor_from_pointer() raises:
    var new_pointer = UnsafePointer[Int8].alloc(5)
    new_pointer[0] = 0
    new_pointer[1] = 1
    new_pointer[2] = 2
    # rest is not initialized

    var some_array = Array[DType.int8](unsafe_pointer=new_pointer, length=3)
    assert_equal(some_array[0], 0)
    assert_equal(some_array[1], 1)
    assert_equal(some_array[2], 2)
    assert_equal(len(some_array), 3)


fn test_constructor_from_other_list_through_pointer() raises:
    # TODO
    # var initial_array = Array[DType.](0, 1, 2)
    # # we do a backup of the size and capacity because
    # # the list attributes will be invalid after the steal_data call
    # var size = len(initial_array)
    # var capacity = initial_array.capacity
    # var some_array = Array[DType.int8](
    #     unsafe_pointer=initial_array.steal_data(), size=size
    # )
    # assert_equal(some_array[0], 0)
    # assert_equal(some_array[1], 1)
    # assert_equal(some_array[2], 2)
    # assert_equal(len(some_list), size)
    # assert_equal(some_list.capacity, capacity)
    pass


fn test_array_to_string() raises:
    var my_array = Array[DType.int8](1, 2, 3)
    assert_equal(str(my_array), "[1, 2, 3]")

    # TODO: need bin func for StringLiteral
    # var a = bin("a")
    # var b = bin("b")
    # var c = bin("c")
    # var foo = bin("foo")

    # var my_array4 = Array[UInt64]("a", "b", "c", "foo")
    # assert_equal(
    #     str(my_array4), "['" + a + "', '" + b + "', '" + c + "', '" + foo + " ']"
    # )


fn test_array_count() raises:
    var arr1 = Array[DType.int8, 10](1, 2, 3, 2, 5, 6, 7, 8, 9, 10)
    assert_equal(1, arr1.count(1))
    assert_equal(2, arr1.count(2))
    assert_equal(0, arr1.count(4))
    assert_equal(0, arr1.count(0))

    var arr2 = Array[DType.int8]()
    assert_equal(0, arr2.count(1))


fn test_array_concat() raises:
    var a = Array[DType.int8](1, 2, 3)
    var b = Array[DType.int8](4, 5, 6)
    # TODO: once lazy evaluation issue is solved
    # var c = a.append(b)
    # assert_equal(len(c), 6)
    # check that original values aren't modified
    assert_equal(len(a), 3)
    assert_equal(len(b), 3)
    # assert_equal(str(c), "[1, 2, 3, 4, 5, 6]")

    # a.extend(b)
    # assert_equal(len(a), 6)
    # assert_equal(str(a), "[1, 2, 3, 4, 5, 6]")
    # assert_equal(len(b), 3)

    # a = Array[DType.int8](1, 2, 3)
    # a.extend(b^)
    # assert_equal(len(a), 6)
    # assert_equal(str(a), "[1, 2, 3, 4, 5, 6]")

    # var d = Array[DType.int8](1, 2, 3)
    # var e = Array[DType.int8](4, 5, 6)
    # var f = d.append(e^)
    # assert_equal(len(f), 6)
    # assert_equal(str(f), "[1, 2, 3, 4, 5, 6]")

    # var l = Array[DType.int8](1, 2, 3)
    # l.extend(Array[DType.int8]())
    # assert_equal(len(l), 3)


fn test_array_contains() raises:
    var x = Array[DType.int8, 3](1, 2, 3)
    assert_false(0 in x)
    assert_true(1 in x)
    assert_false(4 in x)


fn test_indexing() raises:
    var l = Array[DType.int8](1, 2, 3)
    assert_equal(l[int(1)], 2)
    assert_equal(l[False], 1)
    assert_equal(l[True], 2)
    assert_equal(l[2], 3)


fn test_array_unsafe_set_and_get() raises:
    var arr = Array[DType.int8, 5]()

    for i in range(5):
        arr.unsafe_set(i, i)
        arr.capacity_left -= 1

    assert_equal(5, len(arr))
    assert_equal(0, arr.unsafe_get(0))
    assert_equal(1, arr.unsafe_get(1))
    assert_equal(2, arr.unsafe_get(2))
    assert_equal(3, arr.unsafe_get(3))
    assert_equal(4, arr.unsafe_get(4))

    arr[2] = -2
    assert_equal(-2, arr.unsafe_get(2))

    arr.clear()
    arr.unsafe_set(0, 2)
    assert_equal(2, arr.unsafe_get(0))
    assert_equal(0, len(arr))


fn test_min() raises:
    # TODO
    pass


fn test_max() raises:
    # TODO
    pass


fn test_dot() raises:
    # TODO
    pass


fn test_array_add() raises:
    # TODO
    pass


fn test_array_sub() raises:
    # TODO
    pass


fn test_cos() raises:
    # TODO
    pass


fn test_theta() raises:
    # TODO
    pass


fn test_cross() raises:
    # TODO
    pass


fn test_apply() raises:
    # TODO
    pass


fn test_reversed() raises:
    # TODO
    pass


fn test_filter() raises:
    # TODO
    pass


fn main() raises:
    test_array()
    test_array_with_default()
    test_indexing_vec()
    test_mojo_issue_698()
    test_array_to_bool_conversion()
    test_array_pop()
    test_array_variadic_constructor()
    test_array_insert()
    test_array_index()
    test_array_extend()
    test_array_iter()
    test_array_span()
    test_array_boolable()
    test_constructor_from_pointer()
    test_constructor_from_other_list_through_pointer()
    test_array_to_string()
    test_array_count()
    test_array_concat()
    test_array_contains()
    test_indexing()
    test_array_unsafe_set_and_get()
    test_array_iter_not_mutable()
    test_array_broadcast_ops()
    test_min()
    test_max()
    test_dot()
    test_array_add()
    test_array_sub()
    test_cos()
    test_theta()
    test_cross()
    test_apply()
    test_reversed()
    test_filter()
