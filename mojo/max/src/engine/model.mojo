# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Defines the `Model` type that holds a model ready for execution.
"""

from max._utils import call_dylib_func, exchange
from max._driver import (
    AnyMemory,
    AnyTensor,
    AnyMojoValue,
    Device,
    DeviceMemory,
    DeviceTensor,
    _steal_device_memory_impl_ptr,
)
from sys.ffi import DLHandle

from ._model_impl import CModel
from ._compilation import CompiledModel
from ._context import CRuntimeContext
from ._status import Status, CStatus
from .tensor_map import CTensorMap

from max.tensor import Tensor, TensorSpec


@value
struct Model:
    """Represents a model that's loaded and ready for execution.

    Do not instantiate this object directly. Instead, create it with
    [`InferenceSession.load()`](/max/api/mojo/engine/session/InferenceSession#load).
    For example:

    ```mojo
    var session = engine.InferenceSession()
    var model = session.load("bert-base-uncased")
    ```

    Then you can run inference by passing your inputs to [`execute()`](#execute)
    as a NumPy array, a
    [`TensorMap`](/max/api/mojo/engine/tensor_map/TensorMap), or one
    of the other [tensor types](/max/api/mojo/engine/tensor).
    """

    var _ctx: CRuntimeContext
    var _ptr: CModel
    var _lib: DLHandle
    var _session: InferenceSession
    var _compiled_model: CompiledModel
    var _device: Device

    alias _InitModelFnName = "M_initModel"
    alias _ExecuteFnName = "M_executeModelSync"

    fn __moveinit__(inout self, owned existing: Self):
        """Move initializer for model.

        Args:
          existing: Model to move.
        """
        self._ctx = existing._ctx
        self._ptr = exchange[CModel](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session^
        self._compiled_model = existing._compiled_model^
        self._device = existing._device^

    fn _add_to_output_list(
        self,
        list: UnsafePointer[NoneType],
        device_memory_ptr: UnsafePointer[NoneType],
        spec_ptr: UnsafePointer[NoneType],
    ) raises:
        var spec = spec_ptr.bitcast[TensorSpec]()[]
        var device_memory = DeviceTensor(
            DeviceMemory(device_memory_ptr, spec.bytecount(), self._device),
            spec,
        )
        var typed_list = list.bitcast[List[AnyTensor]]()
        typed_list[].append(device_memory^)

    fn execute(self, inputs: TensorMap) raises -> TensorMap:
        """Execute model with given inputs.

        Args:
          inputs: A tensor map with input names as keys and inputs as values.

        Returns:
            A TensorMap with output names as keys.
        """
        var status = Status(self._lib)
        var outputs = call_dylib_func[CTensorMap](
            self._lib,
            Self._ExecuteFnName,
            self._ctx,
            self._ptr,
            inputs._borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()
        return TensorMap(outputs, self._lib, self._session)

    fn execute(self, inputs: PythonObject) raises -> TensorMap:
        """Execute model with given inputs.

        Args:
            inputs:
                Inputs as a Python object, which must be a dictionary with
                string keys (matching input names) and NumPy array values.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = self._session.new_tensor_map()
        for py_pair in inputs.items():
            input_map.borrow(str(py_pair[0]), EngineNumpyView(py_pair[1]))
        return self.execute(input_map)

    fn execute(self, *inputs: NamedTensor) raises -> TensorMap:
        """Execute model with given inputs.

        Args:
          inputs: A variadic list of NamedTensor values.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session)

        for named_tensor in inputs:
            input_map.borrow(named_tensor[].name, named_tensor[]._view)

        var result = self.execute(input_map)

        return result^

    fn execute(
        self, *inputs: Tuple[StringLiteral, EngineNumpyView]
    ) raises -> TensorMap:
        """Execute model with given inputs.

        Args:
          inputs: A variadic list of tuples with first element of tuple is
                  input name and second element is non owning view of a Numpy
                  array.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session)
        for pair in inputs:
            input_map.borrow(pair[][0], pair[][1])
        return self.execute(input_map)

    fn execute[
        type: DType
    ](self, name: String, input: Tensor[type]) raises -> TensorMap:
        """Execute model with given input.

        Parameters:
            type: DType of input tensor.

        Args:
          name: Name of the input tensor.
          input: Input tensor to the model.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session)
        input_map.borrow(name, input)
        return self.execute(input_map)

    fn execute(
        self, name: String, inout input: PythonObject
    ) raises -> TensorMap:
        """Execute model with given input.

        Args:
          name: Name of the input tensor.
          input: Input to the model as numpy array.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session)
        input_map.borrow(name, EngineNumpyView(input))
        return self.execute(input_map)

    fn execute[
        type1: DType, type2: DType
    ](
        self,
        name1: String,
        input1: Tensor[type1],
        name2: String,
        input2: Tensor[type2],
    ) raises -> TensorMap:
        """Execute model with given inputs.

        Parameters:
            type1: DType of first input tensor.
            type2: DType of second input tensor.

        Args:
          name1: Name of the first input tensor.
          input1: First Input tensor to the model.
          name2: Name of the second input tensor.
          input2: Second Input tensor to the model.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session)
        input_map.borrow(name1, input1)
        input_map.borrow(name2, input2)
        return self.execute(input_map)

    fn execute(
        self,
        name1: String,
        inout input1: PythonObject,
        name2: String,
        inout input2: PythonObject,
    ) raises -> TensorMap:
        """Execute model with given inputs.

        Args:
          name1: Name of the first input tensor.
          input1: First Input to the model as numpy array.
          name2: Name of the second input tensor.
          input2: Second Input to the model as numpy array.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session)
        input_map.borrow(name1, EngineNumpyView(input1))
        input_map.borrow(name2, EngineNumpyView(input2))
        return self.execute(input_map)

    fn execute[
        type1: DType, type2: DType, type3: DType
    ](
        self,
        name1: String,
        input1: Tensor[type1],
        name2: String,
        input2: Tensor[type2],
        name3: String,
        input3: Tensor[type3],
    ) raises -> TensorMap:
        """Execute model with given inputs.

        Parameters:
            type1: DType of first input tensor.
            type2: DType of second input tensor.
            type3: DType of third input tensor.

        Args:
          name1: Name of the first input tensor.
          input1: First Input tensor to the model.
          name2: Name of the second input tensor.
          input2: Second Input tensor to the model.
          name3: Name of the third input tensor.
          input3: Third Input tensor to the model.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session)
        input_map.borrow(name1, input1)
        input_map.borrow(name2, input2)
        input_map.borrow(name3, input3)
        return self.execute(input_map)

    fn execute(
        self,
        name1: String,
        inout input1: PythonObject,
        name2: String,
        inout input2: PythonObject,
        name3: String,
        inout input3: PythonObject,
    ) raises -> TensorMap:
        """Execute model with given inputs.

        Args:
          name1: Name of the first input tensor.
          input1: First Input to the model as numpy array.
          name2: Name of the second input tensor.
          input2: Second Input to the model as numpy array.
          name3: Name of the third input tensor.
          input3: Third Input to the model as numpy array.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session)
        input_map.borrow(name1, EngineNumpyView(input1))
        input_map.borrow(name2, EngineNumpyView(input2))
        input_map.borrow(name3, EngineNumpyView(input3))
        return self.execute(input_map)

    fn _execute(self, owned *inputs: AnyMemory) raises -> List[AnyTensor]:
        """Execute model with the given inputs.

        Arguments:
            inputs: Inputs can either be tensors or mojo objects for opaque
            types defined in the graph, which may be located on any Device. If
            inputs are tensors, API will automatically copy the input tensors
            to the Device set in the InferenceSession's SessionConfig.
        Returns:
            A list of output tensors, which are located on the Device set in the
            InferenceSession's SessionConfig.
        """
        var on_device_inputs = List[AnyTensor]()
        var values_impl = List[AnyMojoValue.c_type]()
        for input in inputs:
            if input[].is_tensor():
                var tensor = input[].take_tensor()
                if tensor._device == self._session._ptr[].device:
                    on_device_inputs.append(tensor^)
                else:
                    var input_dt = (
                        tensor.to_device_tensor().copy_to(
                            self._session._ptr[].device
                        )
                    )
                    on_device_inputs.append(input_dt)
            else:
                values_impl.append(input[].take_value().release())

        var on_device_inputs_impl = List[UnsafePointer[NoneType]]()
        var inputs_spec = List[TensorSpec]()
        for input in on_device_inputs:
            inputs_spec.append(input[]._spec)
            on_device_inputs_impl.append(
                _steal_device_memory_impl_ptr(input[].take())
            )

        alias execute_func_name = "M_executeDeviceTensor"

        # TODO: This should return AnyMemory to support returning
        # opaque types.
        var output_list = List[AnyTensor]()
        var output_list_address = UnsafePointer.address_of(output_list)
        var status = Status(self._lib)

        # FIXME: This has too many arguments now. Refactor this into something
        # cleaner.
        var execute_func = self._lib.get_function[
            fn (
                CRuntimeContext,
                CModel,
                UnsafePointer[
                    UnsafePointer[NoneType]
                ],  # UnsafePointer to input tensors
                UnsafePointer[TensorSpec],  # UnsafePointer to input specs
                Int,  # Number of input specs
                UnsafePointer[
                    AnyMojoValue.c_type
                ],  # UnsafePointer to opaque mojo values
                Int,  # Number of opaque mojo values
                __type_of(
                    Self._add_to_output_list
                ),  # Function pointer to add output
                UnsafePointer[Self],  # Calling context
                UnsafePointer[NoneType],  # UnsafePointer to output list
                CStatus,
            ) -> Int
        ](execute_func_name)
        var output_count = execute_func(
            self._ctx,
            self._ptr,
            on_device_inputs_impl.unsafe_ptr(),
            inputs_spec.unsafe_ptr(),
            len(on_device_inputs_impl),
            values_impl.unsafe_ptr(),
            len(values_impl),
            Self._add_to_output_list,
            UnsafePointer.address_of(self),
            output_list_address.bitcast[NoneType](),
            status.borrow_ptr(),
        )

        if status:
            raise str(status)

        if len(output_list) != output_count:
            raise "internal error: mismatch on output count during ffi"

        # Make sure inputs are alive
        _ = on_device_inputs_impl^
        _ = inputs_spec^
        _ = values_impl^
        return output_list

    fn num_model_inputs(self) raises -> Int:
        """Gets the number of inputs of the model.

        Returns:
            Number of inputs of model.
        """

        return self._compiled_model.num_model_inputs()

    fn get_model_input_names(self) raises -> List[String]:
        """Gets the names of model inputs.

        Returns:
            Input names of the model.
        """

        return self._compiled_model.get_model_input_names()

    fn num_model_outputs(self) raises -> Int:
        """Gets the number of outputs of the model.

        Returns:
            Number of model outputs.
        """

        return self._compiled_model.num_model_outputs()

    fn get_model_output_names(self) raises -> List[String]:
        """Gets the names of model outputs.

        Returns:
            Output names of the model.
        """
        return self._compiled_model.get_model_output_names()

    fn get_model_input_metadata(self) raises -> List[EngineTensorSpec]:
        """Get metadata about the model's input tensors, as a list of
        [`EngineTensorSpec`](/max/api/mojo/engine/tensor_spec/EngineTensorSpec)
        objects.

        Returns:
            Metadata list of the model's input tensors.
        """
        return self._compiled_model.get_model_input_metadata()

    fn get_model_output_metadata(self) raises -> List[EngineTensorSpec]:
        """Get metadata about the model's output tensors, as a list of
        [`EngineTensorSpec`](/max/api/mojo/engine/tensor_spec/EngineTensorSpec)
        objects.

        Returns:
            Metadata list of the model's output tensors.
        """
        return self._compiled_model.get_model_output_metadata()

    fn __del__(owned self):
        """Destructor for Model."""
        self._ptr.free(self._lib)
        _ = self._compiled_model^
        _ = self._session^
