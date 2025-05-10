# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements type erased generic tensor and memory types.

For example you can use AnyTensor if you don't know the dtype of tensor in
advance or if you don't know the tensor is DeviceTensor or Tensor:

```mojo
from max.driver import Tensor, AnyTensor

@value
struct Container:
    var _tensor: AnyTensor

def main():
    tensor = Tensor[DType.float32, rank=1]((1,))
    container = Container(tensor^)
```
"""
from .device_memory import DeviceMemory
from .tensor import Tensor
from .device import Device, cpu_device
from max.tensor import TensorSpec
from max._utils import exchange
from collections import Optional
from utils import Variant
from utils._serialize import _serialize
from sys import is_x86, external_call, sizeof, alignof


struct AnyTensor:
    """A type erased tensor representation that is useful
    for situations where we need variadics of tensors."""

    var _data: UnsafePointer[UInt8]
    var _spec: TensorSpec
    var _device: Device
    var _name: Optional[String]
    var _device_memory_impl_ptr: UnsafePointer[NoneType]

    fn __init__(inout self) raises:
        """Default constructor for AnyTensor."""
        self._device = Device()
        self._spec = TensorSpec(DType.uint8, 0)
        self._name = None
        self._data = UnsafePointer[UInt8]()
        self._device_memory_impl_ptr = UnsafePointer[NoneType]()

    fn __init__(inout self, owned device_tensor: DeviceTensor):
        """Creates AnyTensor from a DeviceTensor.

        Args:
            device_tensor: DeviceTensor to construct AnyTensor from.
        """
        self._device = device_tensor.device()
        self._spec = device_tensor.spec
        self._name = device_tensor.name()
        self._data = device_tensor.unsafe_ptr()
        var tmp = device_tensor^
        var tmp_dm = tmp._storage^
        tmp._storage = DeviceMemory()
        self._device_memory_impl_ptr = tmp_dm^._steal_impl_ptr()

    @doc_private
    fn __copyinit__(inout self, existing: Self):
        constrained[False, "AnyTensor is non-copyable"]()
        self._device = existing._device
        self._spec = existing._spec
        self._name = existing._name
        self._data = existing._data
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn __moveinit__(inout self, owned existing: Self):
        """Move constructor for AnyTensor.

        Args:
            existing: Instance to move from.
        """
        self._device = existing._device^
        self._spec = existing._spec^
        self._name = existing._name^
        self._data = existing._data
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        """Creates AnyTensor from a Tensor.

        Args:
            tensor: Tensor to construct AnyTensor from.
        """
        self = Self(tensor^.to_device_tensor())

    fn get_rank(self) -> Int:
        """Gets rank of the tensor.

        Returns:
            Rank of the tensor.
        """
        return self._spec.rank()

    fn spec(self) -> TensorSpec:
        """Gets the spec of the tensor.

        Returns:
            Spec of the tensor.
        """
        return self._spec

    fn _steal_ptr(owned self) -> UnsafePointer[UInt8]:
        var ptr = self._data
        self._data = UnsafePointer[UInt8]()
        return ptr

    fn to_device_tensor(owned self) raises -> DeviceTensor:
        """Consumes this AnyTensor and converts it into a device tensor.

        Returns:
            DeviceTensor representation of AnyTensor.
        """
        var spec = self._spec
        return DeviceTensor(DeviceMemory(self^), spec)

    fn to_tensor[
        type: DType, rank: Int
    ](owned self) raises -> Tensor[type, rank]:
        """Consumes this anytensor and convert it into a tensor.

        Parameters:
            type: Type of tensor.
            rank: Rank of tensor.

        Returns:
            Tensor representation of AnyTensor.
        """
        return self^.to_device_tensor().to_tensor[type, rank]()

    fn take(inout self) raises -> Self:
        """The returned value takes self's resources and replaces them with default
        initialized values.

        Returns:
            Newly constructed anytensor that takes storage from this.
        """
        var tmp = Self()
        swap(self, tmp)
        return tmp

    fn __del__(owned self):
        """Destructor for AnyTensor."""
        _ = DeviceMemory(
            self._device_memory_impl_ptr, self._spec.bytecount(), self._device
        )

    @no_inline
    fn __str__(self) -> String:
        """Gets the tensor as a string.

        Returns:
          A compact string of the tensor.
        """

        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        """
        Formats this Tensor to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        writer.write("Tensor(")

        @parameter
        fn write_dtype_and_shape():
            writer.write("dtype=")
            writer.write(self._spec.dtype())
            writer.write(", ")
            writer.write("shape=")
            for i in range(self.get_rank()):
                if i > 0:
                    writer.write("x")
                writer.write(self._spec.shape[i])

        var device_str = str(self._device)
        if "cpu" not in device_str:
            writer.write("<Unable to print device tensor>, ")
            writer.write(device_str)
            writer.write(", ")
            write_dtype_and_shape()
            writer.write(")")
            return

        @parameter
        fn serialize[T: Formattable](val: T):
            writer.write(val)

        @parameter
        fn dispatcher[dt: DType]():
            var shape = List[Int, hint_trivial_type=True]()
            for i in range(self.get_rank()):
                shape.append(self._spec.shape[i])
            _serialize[serialize_fn=serialize, serialize_end_line=False](
                self._data.bitcast[SIMD[dt, 1]](), shape
            )

        var type = self._spec.dtype()
        try:

            @parameter
            if is_x86():
                type._dispatch_custom[
                    dispatcher,
                    DType.bool,
                    DType.int8,
                    DType.uint8,
                    DType.int16,
                    DType.uint16,
                    DType.int32,
                    DType.uint32,
                    DType.int64,
                    DType.uint64,
                    DType.bfloat16,
                    DType.float16,
                    DType.float32,
                    DType.float64,
                    DType.index,
                ]()
            else:
                # Exclude DType.bfloat16, which is not supported on ARM
                # architectures.
                type._dispatch_custom[
                    dispatcher,
                    DType.bool,
                    DType.int8,
                    DType.uint8,
                    DType.int16,
                    DType.uint16,
                    DType.int32,
                    DType.uint32,
                    DType.int64,
                    DType.uint64,
                    DType.float16,
                    DType.float32,
                    DType.float64,
                    DType.index,
                ]()
        except err:
            writer.write("<Error occured when formatting dtype>, ")
            write_dtype_and_shape()

        writer.write(")")


@value
@register_passable("trivial")
struct _CMojoValue:
    var _ptr: UnsafePointer[NoneType]

    alias _destroy_func_type = fn (UnsafePointer[NoneType]) -> None
    var _destroy_func: Self._destroy_func_type

    fn __init__(inout self):
        self._ptr = UnsafePointer[NoneType]()
        self._destroy_func = Self._destroy_pointee_wrapper[NoneType]

    fn __init__[T: Movable](inout self, ptr: UnsafePointer[T]):
        self._ptr = ptr.bitcast[NoneType]()
        self._destroy_func = Self._destroy_pointee_wrapper[T]

    @staticmethod
    fn _destroy_pointee_wrapper[T: AnyType](ptr: UnsafePointer[NoneType]):
        ptr.bitcast[T]().destroy_pointee()

    @staticmethod
    fn _no_op_destructor[T: AnyType](ptr: UnsafePointer[NoneType]):
        pass

    @staticmethod
    fn _free(ptr: UnsafePointer[NoneType]):
        external_call["KGEN_CompilerRT_MojoValueFreeBuffer", NoneType](ptr)

    fn destroy(self):
        if self._ptr:
            self._destroy_func(self._ptr)
            self._free(self._ptr)


struct AnyMojoValue:
    """Type erased representation of a mojo object. This is useful for passing
    opaque type as input for graph executution.

    CAUTION: Experimental API.
    """

    """Internal representation of Mojo object."""
    alias c_type = _CMojoValue

    var _impl: Self.c_type

    fn __init__(inout self):
        """Default constructor for MojoValue."""
        self._impl = _CMojoValue()

    @doc_private
    fn __init__(inout self, impl: _CMojoValue):
        self._impl = impl

    fn __init__[T: Movable](inout self, owned val: T):
        """Creates Type erased Mojo Value from T.

        Args:
            val: Object to type erase.
        """
        var ptr = external_call[
            "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[T]
        ](sizeof[T](), alignof[T]())
        ptr.init_pointee_move(val^)
        self._impl = _CMojoValue(ptr)

    @doc_private
    fn __copyinit__(inout self, existing: Self):
        constrained[False, "AnyMojoValue is not copyable"]()
        self._impl = existing._impl

    fn __moveinit__(inout self, owned existing: Self):
        """Move constructor for AnyMojoValue.

        Args:
            existing: Instance to move from.
        """
        self._impl = existing._impl

    fn take(inout self) -> Self:
        """Returns the current value and initializes this object to default
        state.

        Returns:
            An instance of AnyMojoValue.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp^

    @doc_private
    fn release(owned self) -> Self.c_type:
        """Release the underlying Mojo Value pointer. Caller is responsible for
        destroying the object."""
        var impl = exchange(self._impl, _CMojoValue())
        return impl

    fn to[T: Movable](owned self) -> T:
        """Consume this object and produces an instance of T. This doesn't do
        any type check and assumes this AnyMojoValue was created from T.

        Returns:
            Instance of type T.
        """
        var value = self._impl._ptr.bitcast[T]().take_pointee()
        self._impl._destroy_func = _CMojoValue._no_op_destructor[T]
        return value^

    fn __del__(owned self):
        """Destructor for AnyMojoValue."""
        self._impl.destroy()


@value
struct AnyMemory:
    """A generic representation which can either be a Driver Tensor or Mojo object.
    """

    var _value: Variant[AnyTensor, AnyMojoValue]

    fn __init__(inout self):
        "Default constructor for AnyMemory."
        self._value = AnyMojoValue()

    fn __init__(inout self, owned device_tensor: DeviceTensor):
        """Creates AnyMemory from a DeviceTensor.

        Args:
            device_tensor: DeviceTensor to construct AnyMemory from.
        """
        self._value = AnyTensor(device_tensor^)

    fn __init__[
        type: DType, rank: Int
    ](inout self, owned tensor: Tensor[type, rank]) raises:
        """Creates AnyMemory from a Tensor.

        Args:
            tensor: Tensor to construct AnyMemory from.
        """
        self._value = AnyTensor(tensor^)

    fn __init__(inout self, owned tensor: AnyTensor):
        """Creates AnyMemory from a AnyTensor.

        Args:
            tensor: AnyTensor to construct AnyMemory from.
        """
        self._value = tensor^

    fn __init__(inout self, owned value: AnyMojoValue):
        """Creates AnyMemory from AnyMojoValue.

        Args:
            value: AnyMojoValue to construct AnyMemory from.
        """
        self._value = value^

    fn is_tensor(self) -> Bool:
        """Check whether this contains a tensor.

        Returns:
            True if contains tensor.
        """
        return self._value.isa[AnyTensor]()

    fn take_tensor(inout self) raises -> AnyTensor:
        """Take tensor from object. Further access to this object is
            undefined behavior.

        Returns:
            The tensor inside the memory as AnyTensor.
        """
        return self._value[AnyTensor].take()

    fn take(inout self) -> Self:
        """The returned value takes self's resources and replaces them with
        default initialized values.

        Returns:
            Newly constructed AnyMemory that takes storage from this.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp^

    fn to_device_tensor(owned self) raises -> DeviceTensor:
        """Consume this object and produces and instance of DeviceTensor.
        Only valid if this was created from DeviceTensor.

        Returns:
            DeviceTensor representation of AnyMemory.
        """
        var tmp = self^
        return tmp.take_tensor().to_device_tensor()

    fn to[T: Movable](owned self) -> T:
        """Consume this object and produces an instance of T. This doesn't do
        any type check beyond whether this is a AnyTensor or not,
        and if not assume this was created from T.

        Returns:
            An instance of type T.
        """
        var tmp = self^
        var value = tmp.take_value()
        return value.to[T]()

    fn take_value(inout self) -> AnyMojoValue:
        """Take value from object. Further access to this object is undefined
        behavior.

        Returns:
            The value inside the memory as AnyMojoValue.
        """
        return self._value[AnyMojoValue].take()

    @no_inline
    fn __str__(self) -> String:
        """Gets this value as a string."""
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        """
        Formats the string representation of this value to the provided
        formatter.

        Args:
            writer: The formatter to write to.
        """
        if self._value.isa[AnyTensor]():
            return writer.write(self._value[AnyTensor])
        else:
            # TODO: Implement print for AnyMojoValue.
            return writer.write("AnyMojoValue")
