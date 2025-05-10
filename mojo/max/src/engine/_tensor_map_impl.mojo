# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from max._utils import call_dylib_func, CString

from ._status import Status
from ._tensor_impl import CTensor
from ._value_impl import CValue


fn _destroy_pointee_wrapper[T: AnyType](ptr: UnsafePointer[T]):
    ptr.destroy_pointee()


@value
@register_passable("trivial")
struct CTensorMap:
    """Represents AsyncTensorMap ptr from Engine."""

    var ptr: DTypePointer[DType.invalid]

    alias CopyAsyncTensorMapFnName = "M_copyAsyncTensorMap"
    alias FreeAsyncTensorMapFnName = "M_freeAsyncTensorMap"
    alias BorrowTensorIntoFnName = "M_borrowTensorInto"
    alias BorrowValueIntoFnName = "M_borrowValueInto"
    alias MoveMojoValueIntoFnName = "M_moveMojoValueInto"
    alias GetTensorByNameFromFnName = "M_getTensorByNameFrom"
    alias GetValueByNameFromFnName = "M_getValueByNameFrom"
    alias GetTensorMapSizeFnName = "M_getTensorMapSize"
    alias KeysFnName = "M_tensorMapKeys"

    fn get_tensor_by_name(self, name: CString, lib: DLHandle) raises -> CTensor:
        var status = Status(lib)
        var tensor = call_dylib_func[CTensor](
            lib, Self.GetTensorByNameFromFnName, self, name, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return tensor

    fn get_value_by_name(self, name: CString, lib: DLHandle) raises -> CValue:
        var status = Status(lib)
        var value = call_dylib_func[CValue](
            lib, Self.GetValueByNameFromFnName, self, name, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return value

    fn borrow_tensor_by_name(
        self,
        ptr: DTypePointer[DType.invalid],
        spec: EngineTensorSpec,
        lib: DLHandle,
    ) raises:
        var status = Status(lib)
        call_dylib_func(
            lib,
            Self.BorrowTensorIntoFnName,
            self,
            ptr,
            spec._borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

    fn borrow_value_by_name(
        self,
        name: String,
        ptr: DTypePointer[DType.invalid],
        lib: DLHandle,
    ) raises:
        var status = Status(lib)
        call_dylib_func(
            lib,
            Self.BorrowValueIntoFnName,
            self,
            name.unsafe_ptr(),
            ptr,
            status.borrow_ptr(),
        )
        _ = name
        if status:
            raise status.__str__()

    fn move_mojo_value_by_name[
        T: Movable
    ](self, name: String, owned val: T, lib: DLHandle,) raises:
        """Create a new MojoValue object and store in the tensormap.

        Parameters:
            T: Type of the mojo object.

        Arguments:
            name: Name of the entry in the tensormap.
            val: mojo object stored in the map as a MojoValue.
            lib: dlhandle for the lib
        """

        # Allocate buffer and move val.
        var value_destructor = _destroy_pointee_wrapper[T]
        var data_ptr = external_call[
            "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[T]
        ](sizeof[T](), alignof[T]())
        data_ptr.init_pointee_move(val^)

        # Store the data_ptr and destructor into an AnyAsyncValue.
        var status = Status(lib)
        call_dylib_func(
            lib,
            Self.MoveMojoValueIntoFnName,
            self,
            name.unsafe_ptr(),
            data_ptr,
            value_destructor,
            status.borrow_ptr(),
        )
        _ = name
        if status:
            raise str(status)

    fn keys(self, size_ptr: Pointer[Int64], lib: DLHandle) -> Pointer[CString]:
        return call_dylib_func[Pointer[CString]](
            lib, Self.KeysFnName, self, size_ptr
        )

    fn size(self, lib: DLHandle) raises -> Int:
        var status = Status(lib)
        var size = call_dylib_func[Int](
            lib, Self.GetTensorMapSizeFnName, self, status.borrow_ptr()
        )
        if status:
            raise status.__str__()
        return size

    fn copy(self, lib: DLHandle) -> CTensorMap:
        """
        Copies the AsyncTensorMap ptr. Increases underlying refcount.
        """
        return call_dylib_func[CTensorMap, CTensorMap](
            lib,
            Self.CopyAsyncTensorMapFnName,
            self,
        )

    fn free(self, lib: DLHandle):
        """
        Free the AsyncTensorMap ptr.
        """
        call_dylib_func(lib, Self.FreeAsyncTensorMapFnName, self)
