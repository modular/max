# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
An abstracted instance of model from any framework loaded with Max
Engine InferenceSession and is ready for execution.
"""

from ._model_impl import CModel
from ._compilation import CompiledModel
from ._context import CRuntimeContext
from sys.ffi import DLHandle
from ._utils import call_dylib_func, exchange
from ._status import Status
from .tensor_map import CTensorMap
from sys.intrinsics import _mlirtype_is_eq


@value
struct Model:
    """A loaded model that you can execute.

    Do not instantiate this class directly. Instead, create it with InferenceSession.
    """

    var _ctx: CRuntimeContext
    var _ptr: CModel
    var _lib: DLHandle
    var _session: InferenceSession
    var _compiled_model: CompiledModel

    alias _InitModelFnName = "M_initModel"
    alias _ExecuteFnName = "M_executeModelSync"

    fn __moveinit__(inout self, owned existing: Self):
        """Move initializer for model.

        Args:
          existing: Model to move.
        """
        self._ctx = existing._ctx
        self._ptr = exchange[CModel](
            existing._ptr, DTypePointer[DType.invalid].get_null()
        )
        self._lib = existing._lib
        self._session = existing._session ^
        self._compiled_model = existing._compiled_model ^

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
        return TensorMap(outputs, self._lib, self._session.copy())

    fn _execute_view[
        key_type: AnyRegType, value_type: AnyRegType
    ](
        self, inputs: VariadicList[Tuple[key_type, value_type]]
    ) raises -> TensorMap:
        var input_map = TensorMap(self._ctx, self._lib, self._session.copy())
        for i in range(len(inputs)):
            var pair = inputs[i]

            # TODO: This is likely dead code with the removal of the
            #       StringRef-based API using NamedTensor.
            @parameter
            if _mlirtype_is_eq[key_type, StringRef]():

                @parameter
                if _mlirtype_is_eq[value_type, EngineTensorView]():
                    input_map.borrow(
                        pair.get[0, StringRef](),
                        pair.get[1, EngineTensorView](),
                    )
                elif _mlirtype_is_eq[value_type, EngineNumpyView]():
                    input_map.borrow(
                        pair.get[0, StringRef](), pair.get[1, EngineNumpyView]()
                    )

            elif _mlirtype_is_eq[key_type, StringLiteral]():

                @parameter
                if _mlirtype_is_eq[value_type, EngineTensorView]():
                    input_map.borrow(
                        pair.get[0, StringLiteral](),
                        pair.get[1, EngineTensorView](),
                    )
                elif _mlirtype_is_eq[value_type, EngineNumpyView]():
                    input_map.borrow(
                        pair.get[0, StringLiteral](),
                        pair.get[1, EngineNumpyView](),
                    )
        return self.execute(input_map)

    fn execute(self, *inputs: NamedTensor) raises -> TensorMap:
        """Execute model with given inputs.

        Args:
          inputs: A variadic list of NamedTensor values.

        Returns:
            A TensorMap with output names as keys.
        """
        var input_map = TensorMap(self._ctx, self._lib, self._session.copy())

        for named_tensor in inputs:
            input_map.borrow(named_tensor[].name, named_tensor[]._view)

        var result = self.execute(input_map)

        return result ^

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
        return self._execute_view[StringLiteral, EngineNumpyView](inputs)

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
        var input_map = TensorMap(self._ctx, self._lib, self._session.copy())
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
        var input_map = TensorMap(self._ctx, self._lib, self._session.copy())
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
        var input_map = TensorMap(self._ctx, self._lib, self._session.copy())
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
        var input_map = TensorMap(self._ctx, self._lib, self._session.copy())
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
        var input_map = TensorMap(self._ctx, self._lib, self._session.copy())
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
        var input_map = TensorMap(self._ctx, self._lib, self._session.copy())
        input_map.borrow(name1, EngineNumpyView(input1))
        input_map.borrow(name2, EngineNumpyView(input2))
        input_map.borrow(name3, EngineNumpyView(input3))
        return self.execute(input_map)

    fn num_model_inputs(self) raises -> Int:
        """Gets the number of inputs of the model.

        Returns:
            Number of inputs of model.
        """

        return self._compiled_model.num_model_inputs()

    fn get_model_input_names(self) raises -> DynamicVector[String]:
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

    fn get_model_output_names(self) raises -> DynamicVector[String]:
        """Gets the names of model outputs.

        Returns:
            Output names of the model.
        """
        return self._compiled_model.get_model_output_names()

    fn __del__(owned self):
        """Destructor for Model."""
        self._ptr.free(self._lib)
        _ = self._compiled_model ^
        _ = self._session ^
