# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Implements functionalities useful for giving input to Max Engine
model for execution.
"""
from collections.vector import DynamicVector
from memory.unsafe import bitcast, DTypePointer, Pointer
from utils.list import Dim
from sys.ffi import DLHandle

from tensor import Tensor
from memory._arc import Arc
from python import Python, PythonObject

from .session import InferenceSession
from .tensor_spec import TensorSpec
from ._tensor_impl import _Numpy, CTensor


@value
struct NamedTensor:
    """A named input tensor."""

    var name: String
    """Name of the tensor."""
    var _tensor: Arc[Tensor[DType.invalid]]
    """Reference-counted pointer keeping the original tensor alive."""
    var _view: EngineTensorView

    fn __init__[
        dtype: DType
    ](inout self, owned name: String, owned tensor: Tensor[dtype]):
        """Creates a `NamedTensor` owning the tensor with a reference count.

        Parameters:
            dtype: Data type of the tensor to own.

        Args:
            name: Name of the tensor.
            tensor: Tensor to take ownership of.
        """
        self.name = name ^

        var tensor_arc = Arc(tensor ^)

        # Get an lvalue reference to the heap-allocated Tensor data, and
        # construct an EngineTensorView from that.
        # This is valid because the data owned by an `Arc` in memory does
        # not move memory location.
        self._view = EngineTensorView(
            __get_address_as_lvalue(tensor_arc._data_ptr().value)
        )

        # Store a type-erased owned copy of the tensor. Erase the type so that
        # `NamedTensor` does not have to be generic.
        self._tensor = tensor_arc._bitcast[Tensor[DType.invalid]]()


@value
@register_passable
struct EngineTensorView:
    """A non-owning register_passable view of a tensor
    that does runtime type checking.

    CAUTION: Make sure the source tensor outlives the view.
    """

    var _ptr: Pointer[Tensor[DType.invalid]]
    var _data_ptr: DTypePointer[DType.invalid]
    var _dtype: DType

    fn __init__[type: DType](inout tensor: Tensor[type]) -> Self:
        """Creates a non-owning view of given Tensor.

        Parameters:
            type: DType of the tensor.

        Args:
            tensor: Tensor backing the view.

        Returns:
            An instance of EngineTensorView of given tensor.
        """
        return Self {
            _ptr: bitcast[Tensor[DType.invalid]](
                Pointer[Tensor[type]].address_of(tensor)
            ),
            _data_ptr: bitcast[DType.invalid](tensor.data()),
            _dtype: type,
        }

    fn data[type: DType](self) raises -> DTypePointer[type]:
        """Returns pointer to the start of tensor.

        Parameters:
            type: Expected type of tensor.

        Returns:
            DTypePointer of given type.

        Raises:
            If the given type does not match the type of tensor.
        """
        if type != self._dtype:
            raise String("Expected type: ") + self._dtype.__str__()
        return bitcast[type](self._data_ptr)

    fn data(self) -> DTypePointer[DType.invalid]:
        """Returns type erased pointer to the start of tensor.

        Returns:
            DTypePointer of invalid type.
        """
        return self._data_ptr

    fn spec(self) raises -> TensorSpec:
        """Returns the spec of tensor backing the view.

        Returns:
            Stdlib TensorSpec of the tensor.
        """

        @always_inline
        @parameter
        fn get_spec[ty: DType]() -> TensorSpec:
            return self._get_value[ty]().spec()

        if self._dtype.is_int8():
            return get_spec[DType.int8]()
        if self._dtype.is_int16():
            return get_spec[DType.int16]()
        if self._dtype.is_int32():
            return get_spec[DType.int32]()
        if self._dtype.is_int64():
            return get_spec[DType.int64]()

        if self._dtype.is_uint8():
            return get_spec[DType.uint8]()
        if self._dtype.is_uint16():
            return get_spec[DType.uint16]()
        if self._dtype.is_uint32():
            return get_spec[DType.uint32]()
        if self._dtype.is_uint64():
            return get_spec[DType.uint64]()

        if self._dtype.is_float16():
            return get_spec[DType.float16]()
        if self._dtype.is_float32():
            return get_spec[DType.float32]()
        if self._dtype.is_float64():
            return get_spec[DType.float64]()
        if self._dtype.is_bool():
            return get_spec[DType.bool]()

        raise String("Expected type: ") + self._dtype.__str__()

    @always_inline("nodebug")
    fn _get_value[type: DType](self) -> Tensor[type]:
        return __get_address_as_lvalue(bitcast[Tensor[type]](self._ptr).address)


@value
@register_passable
struct EngineNumpyView:
    """A non-owning register_passable view of a numpy array.

    CAUTION: Make sure the source array outlives the view.
    """

    var _np: _Numpy
    var _ptr: Pointer[PythonObject]

    fn __init__(inout tensor: PythonObject) raises -> Self:
        """Creates a non-owning view of given numpy array.

        Args:
            tensor: Numpy Array backing the view.

        Returns:
            An instance of EngineNumpyView of given array.
        """
        return Self {
            _np: _Numpy(), _ptr: Pointer[PythonObject].address_of(tensor)
        }

    fn data(self) raises -> DTypePointer[DType.invalid]:
        """Returns type erased pointer to the start of numpy array.

        Returns:
            DTypePointer of given type.
        """
        var data_ptr = __get_address_as_lvalue(
            self._ptr.address
        ).ctypes.data.__index__()
        return bitcast[DType.invalid](data_ptr)

    fn dtype(self) raises -> DType:
        """Get DataType of the array backing the view.

        Returns:
            DataType of the array backing the view.
        """
        var self_type = __get_address_as_lvalue(self._ptr.address).dtype
        if self_type == self._np.int8:
            return DType.int8
        if self_type == self._np.int16:
            return DType.int16
        if self_type == self._np.int32:
            return DType.int32
        if self_type == self._np.int64:
            return DType.int64

        if self_type == self._np.uint8:
            return DType.uint8
        if self_type == self._np.uint16:
            return DType.uint16
        if self_type == self._np.uint32:
            return DType.uint32
        if self_type == self._np.uint64:
            return DType.uint64

        if self_type == self._np.float16:
            return DType.float16
        if self_type == self._np.float32:
            return DType.float32
        if self_type == self._np.float64:
            return DType.float64

        raise "Unknown datatype"

    fn spec(self) raises -> TensorSpec:
        """Returns the spec of numpy array backing the view.

        Returns:
            Numpy array spec in format of Stdlib TensorSpec.
        """

        @always_inline
        @parameter
        fn get_spec[ty: DType]() raises -> TensorSpec:
            var shape = DynamicVector[Int]()
            var array_shape = __get_address_as_lvalue(self._ptr.address).shape
            for dim in array_shape:
                shape.push_back(dim.__index__())
            return TensorSpec(ty, shape)

        if self.dtype().is_int8():
            return get_spec[DType.int8]()
        if self.dtype().is_int16():
            return get_spec[DType.int16]()
        if self.dtype().is_int32():
            return get_spec[DType.int32]()
        if self.dtype().is_int64():
            return get_spec[DType.int64]()

        if self.dtype().is_uint8():
            return get_spec[DType.uint8]()
        if self.dtype().is_uint16():
            return get_spec[DType.uint16]()
        if self.dtype().is_uint32():
            return get_spec[DType.uint32]()
        if self.dtype().is_uint64():
            return get_spec[DType.uint64]()

        if self.dtype().is_float16():
            return get_spec[DType.float16]()
        if self.dtype().is_float32():
            return get_spec[DType.float32]()
        if self.dtype().is_float64():
            return get_spec[DType.float64]()
        if self.dtype().is_bool():
            return get_spec[DType.bool]()

        raise String("Expected type: ") + self.dtype().__str__()
