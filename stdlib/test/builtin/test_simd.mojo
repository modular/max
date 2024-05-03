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

from sys import has_neon, simdwidthof

from testing import assert_equal, assert_not_equal, assert_true


def test_cast():
    assert_equal(
        SIMD[DType.bool, 4](False, True, False, True).cast[DType.bool](),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    assert_equal(
        SIMD[DType.bool, 4](False, True, False, True).cast[DType.int32](),
        SIMD[DType.int32, 4](0, 1, 0, 1),
    )

    assert_equal(
        SIMD[DType.float32, 4](0, 1, 0, -12).cast[DType.int32](),
        SIMD[DType.int32, 4](0, 1, 0, -12),
    )

    assert_equal(
        SIMD[DType.float32, 4](0, 1, 0, -12).cast[DType.bool](),
        SIMD[DType.bool, 4](False, True, False, True),
    )


def test_simd_variadic():
    assert_equal(str(SIMD[DType.index, 4](52, 12, 43, 5)), "[52, 12, 43, 5]")


def test_truthy():
    alias dtypes = (
        DType.bool,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float16,
        DType.float32,
        DType.float64,
        DType.index,
        # DType.address  # TODO(29920)
    )

    @parameter
    fn test_dtype[type: DType]() raises:
        # # Scalars of 0-values are false-y, 1-values are truth-y
        assert_equal(False, Scalar[type](False).__bool__())
        assert_equal(True, Scalar[type](True).__bool__())

        # # SIMD vectors are truth-y if _all_ values are truth-y
        assert_equal(True, SIMD[type, 2](True, True).__bool__())

        # # SIMD vectors are false-y if _any_ values are false-y
        assert_equal(False, SIMD[type, 2](False, True).__bool__())
        assert_equal(False, SIMD[type, 2](True, False).__bool__())
        assert_equal(False, SIMD[type, 2](False, False).__bool__())

    @parameter
    fn test_dtype_unrolled[i: Int]() raises:
        alias type = dtypes.get[i, DType]()
        test_dtype[type]()

    unroll[test_dtype_unrolled, dtypes.__len__()]()

    @parameter
    if not has_neon():
        # TODO bfloat16 is not supported on neon #30525
        test_dtype[DType.bfloat16]()


def test_floordiv():
    assert_equal(Int32(2) // Int32(2), 1)
    assert_equal(Int32(2) // Int32(3), 0)
    assert_equal(Int32(2) // Int32(-2), -1)
    assert_equal(Int32(99) // Int32(-2), -50)

    assert_equal(UInt32(2) // UInt32(2), 1)
    assert_equal(UInt32(2) // UInt32(3), 0)

    assert_equal(Float32(2) // Float32(2), 1)
    assert_equal(Float32(2) // Float32(3), 0)
    assert_equal(Float32(2) // Float32(-2), -1)
    assert_equal(Float32(99) // Float32(-2), -50)


def test_mod():
    assert_equal(Int32(99) % Int32(1), 0)
    assert_equal(Int32(99) % Int32(3), 0)
    assert_equal(Int32(99) % Int32(-2), -1)
    assert_equal(Int32(99) % Int32(8), 3)
    assert_equal(Int32(99) % Int32(-8), -5)
    assert_equal(Int32(2) % Int32(-1), 0)
    assert_equal(Int32(2) % Int32(-2), 0)

    assert_equal(UInt32(99) % UInt32(1), 0)
    assert_equal(UInt32(99) % UInt32(3), 0)

    assert_equal(Int(4) % Int32(3), 1)
    assert_equal(
        Int(78) % SIMD[DType.int32, 2](78, 78), SIMD[DType.int32, 2](0, 0)
    )
    assert_equal(
        SIMD[DType.int32, 2](7, 7) % Int(4), SIMD[DType.int32, 2](3, 3)
    )

    var a = SIMD[DType.float32, 16](
        3.1,
        3.1,
        3.1,
        3.1,
        3.1,
        3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        3.1,
        3.1,
        -3.1,
        -3.1,
    )
    var b = SIMD[DType.float32, 16](
        3.2,
        2.2,
        1.2,
        -3.2,
        -2.2,
        -1.2,
        3.2,
        2.2,
        1.2,
        -3.2,
        -2.2,
        -1.2,
        3.1,
        -3.1,
        3.1,
        -3.1,
    )
    assert_equal(
        a % b,
        SIMD[DType.float32, 16](
            3.0999999046325684,
            0.89999985694885254,
            0.69999980926513672,
            -0.10000014305114746,
            -1.3000001907348633,
            -0.5000002384185791,
            0.10000014305114746,
            1.3000001907348633,
            0.5000002384185791,
            -3.0999999046325684,
            -0.89999985694885254,
            -0.69999980926513672,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )


def test_rotate():
    alias simd_width = 4
    alias type = DType.uint32

    assert_equal(
        SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0).rotate_right[1](),
        SIMD[DType.uint16, 8](0, 1, 0, 1, 1, 0, 1, 0),
    )
    assert_equal(
        SIMD[DType.uint32, 8](1, 0, 1, 1, 0, 1, 0, 0).rotate_right[5](),
        SIMD[DType.uint32, 8](1, 0, 1, 0, 0, 1, 0, 1),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[1](),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[2](),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[3](),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-1](),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-2](),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-3](),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-4](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[1](),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[2](),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[3](),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[4](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-1](),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-2](),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-3](),
        SIMD[type, simd_width](1, 1, 0, 1),
    )


def test_shift():
    alias simd_width = 4
    alias type = DType.uint32

    assert_equal(
        SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0).shift_right[1](),
        SIMD[DType.uint16, 8](0, 1, 0, 1, 1, 0, 1, 0),
    )
    assert_equal(
        SIMD[DType.uint32, 8](11, 0, 13, 12, 0, 100, 0, 0).shift_right[5](),
        SIMD[DType.uint32, 8](0, 0, 0, 0, 0, 11, 0, 13),
    )

    assert_equal(
        SIMD[DType.float64, 8](11.1, 0, 13.1, 12.2, 0, 100.4, 0, 0).shift_right[
            5
        ](),
        SIMD[DType.float64, 8](0, 0, 0, 0, 0, 11.1, 0, 13.1),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[1](),
        SIMD[type, simd_width](0, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[2](),
        SIMD[type, simd_width](1, 1, 0, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[3](),
        SIMD[type, simd_width](1, 0, 0, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[4](),
        SIMD[type, simd_width](0, 0, 0, 0),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[1](),
        SIMD[type, simd_width](0, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[2](),
        SIMD[type, simd_width](0, 0, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[3](),
        SIMD[type, simd_width](0, 0, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[4](),
        SIMD[type, simd_width](0, 0, 0, 0),
    )


def test_insert():
    assert_equal(Int32(3).insert(Int32(4)), 4)

    assert_equal(
        SIMD[DType.index, 4](0, 1, 2, 3).insert(SIMD[DType.index, 2](9, 6)),
        SIMD[DType.index, 4](9, 6, 2, 3),
    )

    assert_equal(
        SIMD[DType.index, 4](0, 1, 2, 3).insert[offset=1](
            SIMD[DType.index, 2](9, 6)
        ),
        SIMD[DType.index, 4](0, 9, 6, 3),
    )

    assert_equal(
        SIMD[DType.index, 8](0, 1, 2, 3, 5, 6, 7, 8).insert[offset=4](
            SIMD[DType.index, 4](9, 6, 3, 7)
        ),
        SIMD[DType.index, 8](0, 1, 2, 3, 9, 6, 3, 7),
    )

    assert_equal(
        SIMD[DType.index, 8](0, 1, 2, 3, 5, 6, 7, 8).insert[offset=3](
            SIMD[DType.index, 4](9, 6, 3, 7)
        ),
        SIMD[DType.index, 8](0, 1, 2, 9, 6, 3, 7, 8),
    )


def test_interleave():
    assert_equal(Int32(0).interleave(Int32(1)), SIMD[DType.index, 2](0, 1))

    assert_equal(
        SIMD[DType.index, 2](0, 2).interleave(SIMD[DType.index, 2](1, 3)),
        SIMD[DType.index, 4](0, 1, 2, 3),
    )


def test_deinterleave():
    var tup2 = SIMD[DType.float32, 2](1, 2).deinterleave()
    assert_equal(tup2[0], Float32(1))
    assert_equal(tup2[1], Float32(2))

    var tup4 = SIMD[DType.index, 4](0, 1, 2, 3).deinterleave()
    assert_equal(tup4[0], SIMD[DType.index, 2](0, 2))
    assert_equal(tup4[1], SIMD[DType.index, 2](1, 3))


def test_address():
    assert_equal(Scalar[DType.address](1), 1)
    assert_not_equal(Scalar[DType.address](1), 0)

    assert_true(Bool(Scalar[DType.address](12) > 1))
    assert_true(Bool(Scalar[DType.address](1) < 12))


def test_extract():
    assert_equal(Int64(99).slice[1](), 99)
    assert_equal(Int64(99).slice[1, offset=0](), 99)

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[4](),
        SIMD[DType.index, 4](99, 1, 2, 4),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2, offset=0](),
        SIMD[DType.index, 2](99, 1),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2, offset=2](),
        SIMD[DType.index, 2](2, 4),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2, offset=1](),
        SIMD[DType.index, 2](1, 2),
    )


def test_limits():
    @parameter
    fn test_integral_overflow[type: DType]() raises:
        var max_value = Scalar[type].MAX
        var min_value = Scalar[type].MIN
        assert_equal(max_value + 1, min_value)

    test_integral_overflow[DType.index]()
    test_integral_overflow[DType.int8]()
    test_integral_overflow[DType.uint8]()
    test_integral_overflow[DType.int16]()
    test_integral_overflow[DType.uint16]()
    test_integral_overflow[DType.int32]()
    test_integral_overflow[DType.uint32]()
    test_integral_overflow[DType.int64]()
    test_integral_overflow[DType.uint64]()


def test_add_with_overflow():
    # TODO: replace all the aliases with math.limit.max_finite()
    # and math.limit.min_finite()
    alias uint8_min = 0
    alias uint8_max = 255
    var value_u8: UInt8
    var overflowed_u8: Scalar[DType.bool]
    value_u8, overflowed_u8 = UInt8(uint8_max).add_with_overflow(1)
    assert_equal(value_u8, uint8_min)
    assert_equal(overflowed_u8, True)

    var value_u8x4: SIMD[DType.uint8, 4]
    var overflowed_u8x4: SIMD[DType.bool, 4]
    value_u8x4, overflowed_u8x4 = SIMD[DType.uint8, 4](
        1, uint8_max, 1, uint8_max
    ).add_with_overflow(SIMD[DType.uint8, 4](0, 1, 0, 1))
    assert_equal(value_u8x4, SIMD[DType.uint8, 4](1, uint8_min, 1, uint8_min))
    assert_equal(overflowed_u8x4, SIMD[DType.bool, 4](False, True, False, True))

    alias int8_min = -128
    alias int8_max = 127
    var value_i8: Int8
    var overflowed_i8: Scalar[DType.bool]
    value_i8, overflowed_i8 = Int8(int8_max).add_with_overflow(1)
    assert_equal(value_i8, int8_min)
    assert_equal(overflowed_i8, True)

    var value_i8x4: SIMD[DType.int8, 4]
    var overflowed_i8x4: SIMD[DType.bool, 4]
    value_i8x4, overflowed_i8x4 = SIMD[DType.int8, 4](
        1, int8_max, 1, int8_max
    ).add_with_overflow(SIMD[DType.int8, 4](0, 1, 0, 1))
    assert_equal(value_i8x4, SIMD[DType.int8, 4](1, int8_min, 1, int8_min))
    assert_equal(overflowed_i8x4, SIMD[DType.bool, 4](False, True, False, True))

    alias uint32_min = 0
    alias uint32_max = 4294967295
    var value_u32: UInt32
    var overflowed_u32: Scalar[DType.bool]
    value_u32, overflowed_u32 = UInt32(uint32_max).add_with_overflow(1)
    assert_equal(value_u32, uint32_min)
    assert_equal(overflowed_u32, True)

    var value_u32x4: SIMD[DType.uint32, 4]
    var overflowed_u32x4: SIMD[DType.bool, 4]
    value_u32x4, overflowed_u32x4 = SIMD[DType.uint32, 4](
        1, uint32_max, 1, uint32_max
    ).add_with_overflow(SIMD[DType.uint32, 4](0, 1, 0, 1))
    assert_equal(
        value_u32x4, SIMD[DType.uint32, 4](1, uint32_min, 1, uint32_min)
    )
    assert_equal(
        overflowed_u32x4, SIMD[DType.bool, 4](False, True, False, True)
    )

    alias int32_min = -2147483648
    alias int32_max = 2147483647
    var value_i32: Int32
    var overflowed_i32: Scalar[DType.bool]
    value_i32, overflowed_i32 = Int32(int32_max).add_with_overflow(1)
    assert_equal(value_i32, int32_min)
    assert_equal(overflowed_i32, True)

    var value_i32x4: SIMD[DType.int32, 4]
    var overflowed_i32x4: SIMD[DType.bool, 4]
    value_i32x4, overflowed_i32x4 = SIMD[DType.int32, 4](
        1, int32_max, 1, int32_max
    ).add_with_overflow(SIMD[DType.int32, 4](0, 1, 0, 1))
    assert_equal(value_i32x4, SIMD[DType.int32, 4](1, int32_min, 1, int32_min))
    assert_equal(
        overflowed_i32x4, SIMD[DType.bool, 4](False, True, False, True)
    )


def test_sub_with_overflow():
    # TODO: replace all the aliases with math.limit.max_finite()
    # and math.limit.min_finite()
    alias uint8_min = 0
    alias uint8_max = 255
    var value_u8: UInt8
    var overflowed_u8: Scalar[DType.bool]
    value_u8, overflowed_u8 = UInt8(uint8_min).sub_with_overflow(1)
    assert_equal(value_u8, uint8_max)
    assert_equal(overflowed_u8, True)

    var value_u8x4: SIMD[DType.uint8, 4]
    var overflowed_u8x4: SIMD[DType.bool, 4]
    value_u8x4, overflowed_u8x4 = SIMD[DType.uint8, 4](
        1, uint8_min, 1, uint8_min
    ).sub_with_overflow(SIMD[DType.uint8, 4](0, 1, 0, 1))
    assert_equal(value_u8x4, SIMD[DType.uint8, 4](1, uint8_max, 1, uint8_max))
    assert_equal(overflowed_u8x4, SIMD[DType.bool, 4](False, True, False, True))

    alias int8_min = -128
    alias int8_max = 127
    var value_i8: Int8
    var overflowed_i8: Scalar[DType.bool]
    value_i8, overflowed_i8 = Int8(int8_min).sub_with_overflow(1)
    assert_equal(value_i8, int8_max)
    assert_equal(overflowed_i8, True)

    var value_i8x4: SIMD[DType.int8, 4]
    var overflowed_i8x4: SIMD[DType.bool, 4]
    value_i8x4, overflowed_i8x4 = SIMD[DType.int8, 4](
        1, int8_min, 1, int8_min
    ).sub_with_overflow(SIMD[DType.int8, 4](0, 1, 0, 1))
    assert_equal(value_i8x4, SIMD[DType.int8, 4](1, int8_max, 1, int8_max))
    assert_equal(overflowed_i8x4, SIMD[DType.bool, 4](False, True, False, True))

    alias uint32_min = 0
    alias uint32_max = 4294967295
    var value_u32: UInt32
    var overflowed_u32: Scalar[DType.bool]
    value_u32, overflowed_u32 = UInt32(uint32_min).sub_with_overflow(1)
    assert_equal(value_u32, uint32_max)
    assert_equal(overflowed_u32, True)

    var value_u32x4: SIMD[DType.uint32, 4]
    var overflowed_u32x4: SIMD[DType.bool, 4]
    value_u32x4, overflowed_u32x4 = SIMD[DType.uint32, 4](
        1, uint32_min, 1, uint32_min
    ).sub_with_overflow(SIMD[DType.uint32, 4](0, 1, 0, 1))
    assert_equal(
        value_u32x4, SIMD[DType.uint32, 4](1, uint32_max, 1, uint32_max)
    )
    assert_equal(
        overflowed_u32x4, SIMD[DType.bool, 4](False, True, False, True)
    )

    alias int32_min = -2147483648
    alias int32_max = 2147483647
    var value_i32: Int32
    var overflowed_i32: Scalar[DType.bool]
    value_i32, overflowed_i32 = Int32(int32_min).sub_with_overflow(1)
    assert_equal(value_i32, int32_max)
    assert_equal(overflowed_i32, True)

    var value_i32x4: SIMD[DType.int32, 4]
    var overflowed_i32x4: SIMD[DType.bool, 4]
    value_i32x4, overflowed_i32x4 = SIMD[DType.int32, 4](
        1, int32_min, 1, int32_min
    ).sub_with_overflow(SIMD[DType.int32, 4](0, 1, 0, 1))
    assert_equal(value_i32x4, SIMD[DType.int32, 4](1, int32_max, 1, int32_max))
    assert_equal(
        overflowed_i32x4, SIMD[DType.bool, 4](False, True, False, True)
    )


def test_mul_with_overflow():
    # TODO: replace all the aliases with math.limit.max_finite()
    # and math.limit.min_finite()
    alias uint8_min = 0
    alias uint8_max = 255
    alias uint8_max_x2 = 254
    var value_u8: UInt8
    var overflowed_u8: Scalar[DType.bool]
    value_u8, overflowed_u8 = UInt8(uint8_max).mul_with_overflow(2)
    assert_equal(value_u8, uint8_max_x2)
    assert_equal(overflowed_u8, True)

    var value_u8x4: SIMD[DType.uint8, 4]
    var overflowed_u8x4: SIMD[DType.bool, 4]
    value_u8x4, overflowed_u8x4 = SIMD[DType.uint8, 4](
        1, uint8_max, 1, uint8_max
    ).mul_with_overflow(SIMD[DType.uint8, 4](0, 2, 0, 2))
    assert_equal(
        value_u8x4, SIMD[DType.uint8, 4](0, uint8_max_x2, 0, uint8_max_x2)
    )
    assert_equal(overflowed_u8x4, SIMD[DType.bool, 4](False, True, False, True))

    alias int8_min = -128
    alias int8_max = 127
    alias int8_max_x2 = -2
    var value_i8: Int8
    var overflowed_i8: Scalar[DType.bool]
    value_i8, overflowed_i8 = Int8(int8_max).mul_with_overflow(2)
    assert_equal(value_i8, int8_max_x2)
    assert_equal(overflowed_i8, True)

    var value_i8x4: SIMD[DType.int8, 4]
    var overflowed_i8x4: SIMD[DType.bool, 4]
    value_i8x4, overflowed_i8x4 = SIMD[DType.int8, 4](
        1, int8_max, 1, int8_max
    ).mul_with_overflow(SIMD[DType.int8, 4](0, 2, 0, 2))
    assert_equal(
        value_i8x4, SIMD[DType.int8, 4](0, int8_max_x2, 0, int8_max_x2)
    )
    assert_equal(overflowed_i8x4, SIMD[DType.bool, 4](False, True, False, True))

    alias uint32_min = 0
    alias uint32_max = 4294967295
    alias uint32_max_x2 = 4294967294
    var value_u32: UInt32
    var overflowed_u32: Scalar[DType.bool]
    value_u32, overflowed_u32 = UInt32(uint32_max).mul_with_overflow(2)
    assert_equal(value_u32, uint32_max_x2)
    assert_equal(overflowed_u32, True)

    var value_u32x4: SIMD[DType.uint32, 4]
    var overflowed_u32x4: SIMD[DType.bool, 4]
    value_u32x4, overflowed_u32x4 = SIMD[DType.uint32, 4](
        1, uint32_max, 1, uint32_max
    ).mul_with_overflow(SIMD[DType.uint32, 4](0, 2, 0, 2))
    assert_equal(
        value_u32x4, SIMD[DType.uint32, 4](0, uint32_max_x2, 0, uint32_max_x2)
    )
    assert_equal(
        overflowed_u32x4, SIMD[DType.bool, 4](False, True, False, True)
    )

    alias int32_min = -2147483648
    alias int32_max = 2147483647
    alias int32_max_x2 = -2
    var value_i32: Int32
    var overflowed_i32: Scalar[DType.bool]
    value_i32, overflowed_i32 = Int32(int32_max).mul_with_overflow(2)
    assert_equal(value_i32, int32_max_x2)
    assert_equal(overflowed_i32, True)

    var value_i32x4: SIMD[DType.int32, 4]
    var overflowed_i32x4: SIMD[DType.bool, 4]
    value_i32x4, overflowed_i32x4 = SIMD[DType.int32, 4](
        1, int32_max, 1, int32_max
    ).mul_with_overflow(SIMD[DType.int32, 4](0, 2, 0, 2))
    assert_equal(
        value_i32x4, SIMD[DType.int32, 4](0, int32_max_x2, 0, int32_max_x2)
    )
    assert_equal(
        overflowed_i32x4, SIMD[DType.bool, 4](False, True, False, True)
    )


def main():
    test_cast()
    test_simd_variadic()
    test_truthy()
    test_floordiv()
    test_mod()
    test_rotate()
    test_shift()
    test_insert()
    test_interleave()
    test_deinterleave()
    test_address()
    test_extract()
    test_limits()
    test_add_with_overflow()
    test_sub_with_overflow()
    test_mul_with_overflow()
