# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Represents a sliced view of a tensor. The slice has lifetime same as that of the
tensor from which it is created.

For example, you can create a TensorSlice and use it like this:

```mojo
from max.driver import Tensor

def main():
    tensor = Tensor[DType.float32, rank=3]((1, 2, 3))
    slice_one = tensor[:]
```

"""
from .tensor import Tensor
from max.tensor import StaticTensorSpec, TensorSpec
from max._tensor_utils import TensorLike
from collections import InlineArray
from math import ceil
from max.tensor import TensorSpec
from sys import triple_is_nvidia_cuda
from sys.intrinsics import strided_load, strided_store


@value
@register_passable
struct TensorSlice[
    is_mutable: Bool, //,
    type: DType,
    rank: Int,
    lifetime: AnyLifetime[is_mutable].type,
]:
    """Sliced view of a tensor. This is safe to use even after the last use of
    tensor from which it is created. For creating a slice use the __getitem__
    method defined in tensor.
    """

    alias _ref_type = Reference[Tensor[type, rank], lifetime]
    var _ref: Self._ref_type
    var _unsafe_slice: UnsafeTensorSlice[type, rank]

    @doc_private
    fn __init__(
        inout self, tensor: Self._ref_type, slices: InlineArray[Slice, rank]
    ):
        self = Self(
            tensor,
            UnsafeTensorSlice[type, rank](
                tensor[]._ptr, slices, tensor[]._spec
            ),
        )

    fn static_spec(self) -> StaticTensorSpec[type, rank]:
        """Gets the static spec of the slice.

        Returns:
            Static tensor spec of slice.
        """
        return self._unsafe_slice.get_static_spec()

    fn spec(self) -> TensorSpec:
        """Gets the spec of the slice.

        Returns:
            Spec of slice as TensorSpec.
        """
        return self._unsafe_slice._spec.get_tensor_spec()

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )

        @always_inline
        @parameter
        fn _indexible() -> Bool:
            return triple_is_nvidia_cuda() or "cpu" in str(self._ref[]._device)

        debug_assert[_indexible](
            "Cannot index into non-CPU Tensor from host",
        )
        return self._unsafe_slice[indices]

    @always_inline
    fn __setitem__(self, *indices: Int, val: Scalar[type]):
        """Sets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.
          val: The value to store at the specified indices.
        """

        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )

        @always_inline
        @parameter
        fn _is_cpu() -> Bool:
            return triple_is_nvidia_cuda() or "cpu" in str(self._ref[]._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )
        self._unsafe_slice[indices] = val
