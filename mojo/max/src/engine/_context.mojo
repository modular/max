# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from collections.optional import Optional
from ._utils import call_dylib_func, exchange
from ._status import Status
from sys.param_env import is_defined

alias MODULAR_PRODUCTION = is_defined["MODULAR_PRODUCTION"]()


@value
@register_passable("trivial")
struct AllocatorType:
    var value: Int32
    # This needs to map M_AllocatorType enum on the C API side.
    alias SYSTEM = Int32(0)
    alias CACHING = Int32(1)

    @always_inline("nodebug")
    fn __ne__(self, rhs: Int32) -> Bool:
        return self.value != rhs.value


@value
@register_passable("trivial")
struct CRuntimeConfig:
    var ptr: DTypePointer[DType.invalid]

    alias FreeRuntimeConfigFnName = "M_freeRuntimeConfig"
    alias SetAllocatorTypeFnName = "M_setAllocatorType"
    alias SetDeviceFnName = "M_setDevice"
    alias SetUseExistingRuntimeFnName = "M_setUseExistingRuntime"

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeConfigFnName, self)

    fn set_device(self, borrowed lib: DLHandle, device: String):
        let device_ref = device._strref_dangerous()
        call_dylib_func(lib, Self.SetDeviceFnName, self, device_ref.data, 0)
        device._strref_keepalive()

    fn set_allocator_type(
        self, borrowed lib: DLHandle, allocator_type: AllocatorType
    ):
        call_dylib_func(lib, Self.SetAllocatorTypeFnName, self, allocator_type)

    fn set_use_existing_runtime(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.SetUseExistingRuntimeFnName, self)


@value
@register_passable
struct _Device(Stringable):
    var value: Int

    alias CPU = _Device(0)
    alias CUDA = _Device(1)

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self == _Device.CPU:
            return "cpu"
        return "cuda"


struct RuntimeConfig:
    var ptr: CRuntimeConfig
    var lib: DLHandle

    alias NewRuntimeConfigFnName = "M_newRuntimeConfig"

    fn __init__(
        inout self,
        lib: DLHandle,
        device: _Device,
        allocator_type: AllocatorType = AllocatorType.CACHING,
    ):
        self.ptr = call_dylib_func[CRuntimeConfig](
            lib, Self.NewRuntimeConfigFnName
        )
        # Mojo already has an existing runtime, regardless of the entrypoint.
        # Set the runtime config to reuse this existing runtime, rather than
        # trying to recreate a new one.
        self.ptr.set_use_existing_runtime(lib)
        self.lib = lib

        if allocator_type != AllocatorType.CACHING:
            self.ptr.set_allocator_type(self.lib, allocator_type)

        @parameter
        if MODULAR_PRODUCTION:
            if device != _Device.CPU:
                print(
                    "The device",
                    device,
                    "is not valid. The device must be set to 'cpu'.",
                )
            return
        else:
            if device == _Device.CUDA:
                self.ptr.set_device(self.lib, device)

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CRuntimeConfig](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn borrow_ptr(self) -> CRuntimeConfig:
        """
        Borrow the underlying C ptr.
        """
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)


@value
@register_passable("trivial")
struct CRuntimeContext:
    var ptr: DTypePointer[DType.invalid]

    alias FreeRuntimeContextFnName = "M_freeRuntimeContext"

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeContextFnName, self)


struct RuntimeContext:
    var ptr: CRuntimeContext
    var lib: DLHandle

    alias NewRuntimeContextFnName = "M_newRuntimeContext"

    fn __init__(inout self, owned config: RuntimeConfig, lib: DLHandle):
        let status = Status(lib)
        self.ptr = call_dylib_func[CRuntimeContext](
            lib,
            Self.NewRuntimeContextFnName,
            config.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            print(status.__str__())
            self.ptr = DTypePointer[DType.invalid]()
        _ = config ^
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CRuntimeContext](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib

    fn borrow_ptr(self) -> CRuntimeContext:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
