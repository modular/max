# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s | FileCheck %s

from max.engine import InferenceSession
from tensor import TensorSpec, Tensor, TensorShape
from utils.index import Index


fn test_tensor_map() raises:
    # CHECK-LABEL: ====test_tensor_map
    print("====test_tensor_map")

    var t1 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1

    let session = InferenceSession()
    let map = session.new_tensor_map()

    # CHECK: 0
    print(len(map))

    map.borrow("tensor", t1)

    # CHECK: 1
    print(len(map))
    let t2 = map.get[DType.float32]("tensor")

    # CHECK: 1
    print(len(map))

    let t3 = map.get[DType.float32]("tensor")

    # CHECK: True
    print(t1 == t2)

    # CHECK: True
    print(t1 == t3)


fn test_tensor_map_value() raises:
    # CHECK-LABEL: ====test_tensor_map_value
    print("====test_tensor_map_value")

    var t1 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1

    let session = InferenceSession()
    let map = session.new_tensor_map()
    let t1_value = session.new_borrowed_tensor_value(t1)

    # CHECK: 0
    print(len(map))

    map.borrow("tensor", t1_value)

    # CHECK: 1
    print(len(map))

    let t2 = map.get[DType.float32]("tensor")
    # CHECK: True
    print(t1 == t2)

    let t3_value = map.get_value("tensor")
    let t3 = t3_value.as_tensor_copy[DType.float32]()
    # CHECK: True
    print(t1 == t3)

    _ = map ^
    _ = t1_value ^
    _ = t1 ^


fn main() raises:
    test_tensor_map()
    test_tensor_map_value()
