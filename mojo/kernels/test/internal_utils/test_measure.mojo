# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from buffer import DimList
from internal_utils import TestTensor, assert_with_measure, correlation
from memory import UnsafePointer
from testing import assert_almost_equal


def test_assert_with_custom_measure():
    var t0 = TestTensor[DType.float32, 1](DimList(100), List[Float32](1))
    var t1 = TestTensor[DType.float32, 1](DimList(100), List[Float32](1))

    fn always_zero[
        type: DType
    ](
        lhs: UnsafePointer[Scalar[type]],
        rhs: UnsafePointer[Scalar[type]],
        n: Int,
    ) -> Float64:
        return 0

    assert_with_measure[always_zero](t0, t1)

    _ = t0^
    _ = t1^


fn test_correlation() raises:
    var a = 10
    var b = 10
    var len = a * b
    var u = UnsafePointer[Scalar[DType.float32]].alloc(len)
    var v = UnsafePointer[Scalar[DType.float32]].alloc(len)
    var x = UnsafePointer[Scalar[DType.float32]].alloc(len)
    for i in range(len):
        u.store(i, (0.01 * i).cast[DType.float32]())
        v.store(i, (-0.01 * i).cast[DType.float32]())
    for i in range(a):
        for j in range(b):
            x.store(b * i + j, (0.1 * i + 0.1 * j).cast[DType.float32]())

    assert_almost_equal(1.0, correlation[out_type = DType.float64](u, u, len))
    assert_almost_equal(-1.0, correlation[out_type = DType.float64](u, v, len))
    # +/- 0.773957299203321 is the exactly rounded fp64 answer calculated using mpfr
    assert_almost_equal(
        0.773957299203321, correlation[out_type = DType.float64](u, x, len)
    )
    assert_almost_equal(
        -0.773957299203321, correlation[out_type = DType.float64](v, x, len)
    )
    u.free()
    v.free()
    x.free()


def main():
    test_assert_with_custom_measure()
    test_correlation()
