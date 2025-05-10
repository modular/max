# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# UNSUPPORTED: address
# REQUIRES: numpy
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s | FileCheck %s

from max.engine import EngineNumpyView
from python import Python


fn test_numpy_view() raises:
    # CHECK: test_numpy_view
    print("====test_numpy_view")
    var np = Python.import_module("numpy")

    var n1 = np.array([1, 2, 3]).astype(np.float32)

    var n1_view = EngineNumpyView(n1)

    # CHECK: 3xfloat32
    print(n1_view.spec().__str__())

    _ = n1 ^


fn main() raises:
    test_numpy_view()
