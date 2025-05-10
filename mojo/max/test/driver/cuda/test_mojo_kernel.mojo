# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s %t1

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/cuda-test-mojo-kernel
# RUN: %mojo-build %s -o %t/cuda-test-mojo-kernel
# RUN: %t/cuda-test-mojo-kernel %t2

from pathlib import Path
from sys import stderr

from gpu.host import Dim
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from max.driver import (
    Accelerator,
    Device,
    DeviceTensor,
    DynamicTensor,
    ManagedTensorSlice,
    Tensor,
    accelerator,
    cpu,
)
from max.tensor import TensorShape
from testing import assert_equal


fn vec_add[
    type: DType,
    rank: Int,
](
    in0: Tensor[type, rank].layout_tensor,
    in1: Tensor[type, rank].layout_tensor,
    out: Tensor[type, rank].layout_tensor,
):
    var row = thread_idx.x
    var col = thread_idx.y
    out[row, col] = in0[row, col] + in1[row, col]


def fill(gpu_dev: Device, shape: TensorShape, val: Float32) -> DeviceTensor:
    var host = Tensor[DType.float32, 2](shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            host[i, j] = val
    return host^.to_device_tensor().copy_to(gpu_dev)


def test_vec_add():
    gpu_dev = accelerator()
    cpu_dev = cpu()
    shape = TensorShape(10, 10)
    alias type = DType.float32
    in0 = fill(gpu_dev, shape, 1).to_tensor[type, 2]()
    in1 = fill(gpu_dev, shape, 2).to_tensor[type, 2]()
    out = fill(gpu_dev, shape, 0).to_tensor[type, 2]()

    kernel = Accelerator.compile[vec_add[type, 2]](gpu_dev)
    kernel(
        gpu_dev,
        in0.to_layout_tensor(),
        in1.to_layout_tensor(),
        out.to_layout_tensor(),
        block_dim=Dim(shape[0], shape[1]),
        grid_dim=Dim(1, 1),
    )

    out_host = (
        out^.to_device_tensor().copy_to(cpu_dev).to_tensor[type, 2]()
    )  # copy blocks until the kernel is finished

    # lifetime extension required otherwise in0's last use is before the call to
    # kernel.__call__, even though the destructor is enqueued asynchronously on the stream
    _ = in0
    _ = in1

    for i in range(shape[0]):
        for j in range(shape[1]):
            assert_equal(out_host[i, j], 3)


def main():
    test_vec_add()
