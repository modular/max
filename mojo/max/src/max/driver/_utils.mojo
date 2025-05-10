# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import memcpy, UnsafePointer

from .anytensor import AnyTensor
from max.tensor import Tensor as OldTensor
from .tensor import Tensor
from .device import Device, cpu_device


fn _steal_device_memory_impl_ptr(
    owned memory: AnyTensor,
) raises -> UnsafePointer[NoneType]:
    """This takes `memory` as inout and not owned because it is called on
    References owned by a List (returned by List.__getitem__()).
    """
    var tmp_device_tensor = memory^.to_device_tensor()
    var taken_device_memory = tmp_device_tensor._storage.take()

    var ptr = taken_device_memory^._steal_impl_ptr()
    return ptr


fn _convert_from[
    dtype: DType, rank: Int
](old_tensor: OldTensor[dtype]) raises -> Tensor[dtype, rank]:
    """Converts max.tensor to max.driver.Tensor. This creates tensor on the CPU
       similar to max.Tensor.

    Parameters:
        dtype: DataType of tensor contents.
        rank: Rank of the tensor.
    Args:
        old_tensor: Tensor to copy from.
    Returns:
        Instance of driver tensor with contents copied from old_tensor.
    """
    if old_tensor.rank() != rank:
        raise String("mismatch in rank, expected {} given {}").format(
            old_tensor.rank(), rank
        )

    var dev = cpu_device()

    var new_tensor = Tensor[dtype, rank](old_tensor.spec().shape, dev)
    memcpy(
        dest=new_tensor.unsafe_ptr(),
        src=old_tensor.unsafe_ptr(),
        count=old_tensor.num_elements(),
    )
    return new_tensor
