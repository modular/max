# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional
from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from sys import external_call

from .session import InferenceSession
from ._model_specs import InputTensorNames, OutputTensorNames
from ._status import Status
from ._utils import call_dylib_func, exchange, OwningVector
from ._tensor_spec_impl import CTensorSpec
from tensor import TensorSpec
from collections import List
from ._dtypes import EngineDType
from pathlib import Path


@value
@register_passable("trivial")
struct FrameworkFormat:
    """Enum-like struct indicating the model framework."""

    alias MAXGraph = FrameworkFormat(0)

    var value: UInt8


@value
@register_passable("trivial")
struct ModelSource(CollectionElement):
    """Model source representation that is ABI compatible with the C API's `M_ModelSource`.
    """

    var source: Pointer[NoneType]
    var format: FrameworkFormat


@value
@register_passable("trivial")
struct CCompileConfig:
    """Mojo representation of Engine's CompileConfig pointer.
    This doesn't free the memory on destruction. For memory managed
    option see CompileConfig.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeCompileConfigFnName = "M_freeCompileConfig"
    alias SetModelSourceFnName = "M_setModelSource"
    alias SetModelPathFnName = "M_setModelPath"
    alias ReplaceOpsFnName = "M_useKernelsFrom"
    alias SetDeviceFnName = "M_setDevice"
    alias SetTorchInputSpecsFnName = "M_setTorchInputSpecs"

    fn set_model_source(
        self, model_source: ModelSource, borrowed lib: DLHandle
    ):
        call_dylib_func(lib, Self.SetModelSourceFnName, self, model_source)

    fn set_model_path(self, path: String, borrowed lib: DLHandle):
        """Sets the path of model to compile."""
        var path_strref = path._strref_dangerous()
        call_dylib_func(lib, Self.SetModelPathFnName, self, path_strref.data)
        path._strref_keepalive()

    fn replace_ops(self, path: String, borrowed lib: DLHandle) raises:
        var status = Status(lib)
        var path_strref = path._strref_dangerous()
        call_dylib_func(
            lib, Self.ReplaceOpsFnName, self, path_strref.data, status.ptr
        )
        path._strref_keepalive()
        if status:
            raise Error(status.__str__())

    fn set_torch_input_specs(
        self,
        torch_lib: DLHandle,
        specs_ptr: List[CTorchInputSpec],
    ):
        call_dylib_func(
            torch_lib,
            Self.SetTorchInputSpecsFnName,
            self,
            specs_ptr.data,
            len(specs_ptr),
        )

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompileConfigFnName, self)


@value
@register_passable("trivial")
struct CTorchInputSpec(CollectionElement):
    """C API ABI compatible M_TorchInputSpec."""

    alias ptr_type = Pointer[NoneType]
    var ptr: Self.ptr_type

    alias FreeTorchInputSpecFnName = "M_freeTorchInputSpec"

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeTorchInputSpecFnName, self)


struct TorchInputSpec(Movable):
    alias shape_type = List[Int64]
    var shape: Self.shape_type
    var dtype: DType
    var ptr: CTorchInputSpec
    var torch_lib: DLHandle

    alias NewTorchInputSpecFnName = "M_newTorchInputSpec"

    fn __init__(inout self, spec: TensorSpec, lib: DLHandle):
        var shape = Self.shape_type()
        shape.reserve(spec.rank())
        for i in range(spec.rank()):
            shape.append(spec[i])
        self.shape = shape
        self.dtype = spec.dtype()
        var ptr = call_dylib_func[CTorchInputSpec](
            lib,
            Self.NewTorchInputSpecFnName,
            self.shape.data,
            len(self.shape),
            EngineDType(self.dtype),
        )
        self.ptr = ptr
        self.torch_lib = lib

    fn __init__(
        inout self,
        shape: List[Optional[Int64]],
        dtype: DType,
        lib: DLHandle,
        engine_lib: DLHandle,
    ):
        var converted_shape = Self.shape_type()
        converted_shape.reserve(len(shape))
        for i in range(len(shape)):
            var dim_or = shape[i]
            if dim_or:
                converted_shape.append(dim_or.value())
            else:
                converted_shape.append(
                    CTensorSpec.get_dynamic_dimension_value(engine_lib)
                )

        self.shape = converted_shape ^
        self.dtype = dtype
        var ptr = call_dylib_func[CTorchInputSpec](
            lib,
            Self.NewTorchInputSpecFnName,
            self.shape.data,
            len(self.shape),
            EngineDType(self.dtype),
        )
        self.ptr = ptr
        self.torch_lib = lib

    fn __init__(
        inout self,
        shape: NoneType,
        dtype: DType,
        lib: DLHandle,
        engine_lib: DLHandle,
    ):
        self.shape = Self.shape_type()
        self.dtype = dtype
        var ptr = call_dylib_func[CTorchInputSpec](
            lib,
            Self.NewTorchInputSpecFnName,
            CTorchInputSpec.ptr_type(),
            CTensorSpec.get_dynamic_rank_value(engine_lib),
            EngineDType(self.dtype),
        )
        self.ptr = ptr
        self.torch_lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.shape = existing.shape ^
        self.dtype = existing.dtype
        self.ptr = existing.ptr
        self.torch_lib = existing.torch_lib

    fn __del__(owned self):
        self.ptr.free(self.torch_lib)


struct CompileConfig:
    """Memory managed version of Engine's Compile Config."""

    var ptr: Pointer[CCompileConfig]
    var lib: DLHandle
    var torch_lib: Optional[DLHandle]
    var input_specs: OwningVector[TorchInputSpec]

    alias NewCompileConfigFnName = "M_newCompileConfig"

    fn __init__(inout self, borrowed lib: DLHandle):
        self.ptr = Pointer[CCompileConfig].alloc(1)
        __get_address_as_uninit_lvalue(self.ptr.address) = call_dylib_func[
            CCompileConfig
        ](lib, Self.NewCompileConfigFnName)
        self.lib = lib
        self.input_specs = OwningVector[TorchInputSpec]()
        self.torch_lib = Self._get_torch_lib()

    @staticmethod
    fn _get_torch_lib() -> Optional[DLHandle]:
        # Since we only need to open this library for this case we
        # can lazy load it here.
        var torch_ext_lib_path_str_ptr = external_call[
            "KGEN_CompilerRT_getConfigValue", DTypePointer[DType.int8]
        ]("max.torch_ext_lib")

        if not torch_ext_lib_path_str_ptr:
            return None

        # This transfers ownership of the underlying data buffer allocated in
        # `KGEN_CompilerRT_getConfigValue` so that it can be destroyed by Mojo.
        var torch_ext_lib_path = String._from_bytes(torch_ext_lib_path_str_ptr)

        if not Path(torch_ext_lib_path).exists():
            return None

        return DLHandle(torch_ext_lib_path)

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = existing.ptr
        self.lib = existing.lib
        self.input_specs = existing.input_specs ^
        self.torch_lib = existing.torch_lib

    fn set_model_source(self, model_source: ModelSource):
        self.ptr[].set_model_source(model_source, self.lib)

    fn set_model_path(self, path: String):
        """Sets the path of model to compile."""
        self.ptr[].set_model_path(path, self.lib)

    fn set_replace_ops_path(self, path: String) raises:
        """Replace Modular kernels with user-defined kernels."""
        self.ptr[].replace_ops(path, self.lib)

    fn set_torch_input_specs(self) raises:
        if len(self.input_specs) == 0:
            return

        if not self.torch_lib:
            raise "cannot find torch extension libraries"

        var inner_spec = List[CTorchInputSpec]()
        for i in range(len(self.input_specs)):
            var spec_ptr = self.input_specs.get(i)
            inner_spec.append(spec_ptr[].ptr)
        self.ptr[].set_torch_input_specs(self.torch_lib.value(), inner_spec)

    fn add_input_spec(inout self, spec: TensorSpec):
        self.input_specs.emplace_back(
            TorchInputSpec(spec, self.torch_lib.value())
        )

    fn add_input_spec(
        inout self,
        shape_or: Optional[List[Optional[Int64]]],
        dtype: DType,
    ):
        if not shape_or:
            self.input_specs.emplace_back(
                TorchInputSpec(None, dtype, self.torch_lib.value(), self.lib)
            )
            return
        self.input_specs.emplace_back(
            TorchInputSpec(
                shape_or.value(), dtype, self.torch_lib.value(), self.lib
            )
        )

    fn borrow_ptr(self) -> Pointer[CCompileConfig]:
        return self.ptr

    fn take_ptr(inout self) -> Pointer[CCompileConfig]:
        var ptr = self.ptr
        self.ptr = Pointer[CCompileConfig]()
        return ptr

    fn __del__(owned self):
        if self.torch_lib:
            var torch = self.torch_lib.value()
            torch.close()

        self.ptr[].free(self.lib)
        _ = __get_address_as_owned_value(self.ptr.address)
        self.ptr.free()


@value
@register_passable("trivial")
struct CCompiledModel:
    """Mojo representation of Engine's AsyncCompiledModel pointer.
    Useful for C inter-op.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeCompiledModelFnName = "M_freeCompiledModel"
    alias GetModelInputSpecByNameFnName = "M_getModelInputSpecByName"
    alias GetModelOutputSpecByNameFnName = "M_getModelOutputSpecByName"
    alias GetNumInputsFnName = "M_getNumModelInputs"
    alias GetNumOutputsFnName = "M_getNumModelOutputs"

    fn num_model_inputs(self, borrowed lib: DLHandle) raises -> Int:
        """Gets the number of inputs of the model."""

        var status = Status(lib)
        var num_inputs = call_dylib_func[Int](
            lib, Self.GetNumInputsFnName, self, status.ptr
        )
        if status:
            raise Error(status.__str__())
        return num_inputs

    fn num_model_outputs(self, borrowed lib: DLHandle) raises -> Int:
        """Gets the number of outputs of the model."""

        var status = Status(lib)
        var num_outputs = call_dylib_func[Int](
            lib, Self.GetNumOutputsFnName, self, status.ptr
        )
        if status:
            raise Error(status.__str__())
        return num_outputs

    fn get_model_input_spec_by_name(
        self,
        tensor_name: String,
        borrowed lib: DLHandle,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        """Gets the input spec of the model by name."""
        var status = Status(lib)
        var input_spec = call_dylib_func[CTensorSpec](
            lib,
            Self.GetModelInputSpecByNameFnName,
            self,
            tensor_name._as_ptr(),
            status.ptr,
        )
        if status:
            raise Error(status.__str__())
        return EngineTensorSpec(input_spec, lib, session.copy())

    fn get_model_output_spec_by_name(
        self,
        tensor_name: String,
        borrowed lib: DLHandle,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        """Gets the output spec of the model by name."""
        var status = Status(lib)
        var output_spec = call_dylib_func[CTensorSpec](
            lib,
            Self.GetModelOutputSpecByNameFnName,
            self,
            tensor_name._as_ptr(),
            status.ptr,
        )
        if status:
            raise Error(status.__str__())
        return EngineTensorSpec(output_spec, lib, session.copy())

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompiledModelFnName, self)


@value
struct CompiledModel:
    """Memory managed CompiledModel pointer."""

    var ptr: CCompiledModel
    var lib: DLHandle
    var session: InferenceSession

    alias CompileModelFnName = "M_compileModelSync"

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CCompiledModel](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib
        self.session = existing.session ^

    fn num_model_inputs(self) raises -> Int:
        """Gets the number of inputs of the model."""

        return self.ptr.num_model_inputs(self.lib)

    fn get_model_input_names(self) raises -> List[String]:
        """Gets the names of model inputs."""

        var names = InputTensorNames(
            self.ptr, self.num_model_inputs(), self.lib
        )
        var name_vec = List[String]()
        name_vec.reserve(len(names))
        for i in range(len(names)):
            name_vec.append(names[i])
        return name_vec

    fn num_model_outputs(self) raises -> Int:
        """Gets the number of outputs of the model."""

        return self.ptr.num_model_outputs(self.lib)

    fn get_model_output_names(self) raises -> List[String]:
        """Gets the names of model outputs."""

        var names = OutputTensorNames(
            self.ptr, self.num_model_outputs(), self.lib
        )
        var name_vec = List[String]()
        name_vec.reserve(len(names))
        for i in range(len(names)):
            name_vec.append(names[i])
        return name_vec

    fn get_model_input_metadata(self) raises -> List[EngineTensorSpec]:
        """Get the metadata for inputs of the model."""
        var input_metadata = List[EngineTensorSpec]()
        var input_tensor_names = self.get_model_input_names()
        input_metadata.reserve(len(input_tensor_names))

        for input_tensor_name in input_tensor_names:
            var input_spec = self.ptr.get_model_input_spec_by_name(
                input_tensor_name[], self.lib, self.session.copy()
            )
            input_metadata.append(input_spec ^)
        return input_metadata

    fn get_model_output_metadata(self) raises -> List[EngineTensorSpec]:
        """Get the metadata for outputs of the model."""
        var output_metadata = List[EngineTensorSpec]()
        var output_tensor_names = self.get_model_output_names()
        output_metadata.reserve(len(output_tensor_names))

        for output_tensor_name in output_tensor_names:
            var output_spec = self.ptr.get_model_output_spec_by_name(
                output_tensor_name[], self.lib, self.session.copy()
            )
            output_metadata.append(output_spec ^)
        return output_metadata

    fn borrow_ptr(self) -> CCompiledModel:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
