# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler
from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.matmul import matmul as _matmul
from register import uses_opaque
from runtime.asyncrt import DeviceContextPtr, MojoCallContextPtr
from tensor import ManagedTensorSlice, foreach
from tensor_internal import view_copy_impl
from tensor_internal import (
    simd_store_into_managed_tensor_slice,
    simd_load_from_managed_tensor_slice,
    _input_fusion_hook_impl,
    _output_fusion_hook_impl,
)

from utils import IndexList, StaticTuple
from sys import external_call


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
fn export():
    alias _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    alias _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice
    alias __input_fusion_hook_impl = _input_fusion_hook_impl
    alias __output_fusion_hook_impl = _output_fusion_hook_impl


# ===-----------------------------------------------------------------------===#
# Opaque Reg Types
# ===-----------------------------------------------------------------------===#


@value
@register_passable
struct MyCustomScalarRegSI32:
    var val: Scalar[DType.int32]

    @implicit
    fn __init__(out self, val: Scalar[DType.int32]):
        print("MyCustomScalarRegSI32.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarRegSI32.__del__", self.val)


# It is intentional there are no methods which consume this.
# It is here to support some level of type checking.
@value
@register_passable
struct MyCustomScalarRegF32:
    var val: Scalar[DType.float32]

    @implicit
    fn __init__(out self, val: Scalar[DType.float32]):
        print("MyCustomScalarRegF32.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarRegF32.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_si32_reg", num_dps_outputs=0)
struct OpaqueToCustomScalarSI32Reg:
    @uses_opaque
    @staticmethod
    fn execute(
        x: ManagedTensorSlice[DType.int32, rank=1]
    ) -> MyCustomScalarRegSI32:
        return MyCustomScalarRegSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32_reg")
struct OpaqueAddToTensorSI32Reg:
    @uses_opaque
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[DType.int32, rank=1],
        x: MyCustomScalarRegSI32,
        y: MyCustomScalarRegSI32,
    ):
        out[0] = x.val + y.val


@compiler.register("opaque_add_to_tensor_f32_reg")
struct OpaqueAddToTensorF32:
    @uses_opaque
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[DType.float32, rank=1],
        x: MyCustomScalarRegF32,
        y: MyCustomScalarRegF32,
    ):
        out[0] = x.val + y.val


# ===-----------------------------------------------------------------------===#
# Opaque Mem. Types
# ===-----------------------------------------------------------------------===#


@value
struct MyCustomScalarSI32:
    var val: Scalar[DType.int32]

    @implicit
    fn __init__(out self, val: Scalar[DType.int32]):
        print("MyCustomScalarSI32.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarSI32.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_si32", num_dps_outputs=0)
struct OpaqueToCustomScalarSI32:
    @uses_opaque
    @staticmethod
    fn execute(
        x: ManagedTensorSlice[DType.int32, rank=1]
    ) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32")
struct OpaqueAddToTensorSI32:
    @uses_opaque
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[DType.int32, rank=1],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ):
        out[0] = x.val + y.val


@compiler.register("opaque_add_to_tensor_si32_raises")
struct OpaqueAddToTensorSI32Raises:
    @uses_opaque
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[DType.int32, rank=1],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ) raises:
        out[0] = x.val + y.val


# ===-----------------------------------------------------------------------===#
# Other Kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("imposter_add")
struct Foo:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x.load[width](idx)) + rebind[
                SIMD[z.type, width]
            ](y.load[width](idx))

        foreach[func](z)

    @staticmethod
    fn shape(x: ManagedTensorSlice, y: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


@always_inline
fn toNDBuffer[
    out_dtype: DType, out_rank: Int
](tensor: ManagedTensorSlice) -> NDBuffer[out_dtype, out_rank]:
    # TODO(GEX-734): forward other static params automatically
    return rebind[NDBuffer[out_dtype, out_rank]](
        NDBuffer[tensor.type, tensor.rank](tensor._ptr, tensor.shape())
    )


# c = a @ b, should support CPU and GPU
@compiler.register("imposter_matmul")
struct ImposterMatmul:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        c: ManagedTensorSlice,
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        alias rank = a.rank
        alias a_dtype = a.type
        alias b_dtype = b.type
        alias c_dtype = c.type

        # Convert everything to NDBuffer
        var c_buffer = toNDBuffer[c_dtype, 2](c)
        var a_buffer = toNDBuffer[a_dtype, 2](a)
        var b_buffer = toNDBuffer[b_dtype, 2](b)
        _matmul[
            False,
            False,
            False,
            None,
            saturated_vnni=False,
            single_thread_blocking_override=synchronous,
            target=target,
        ](
            c_buffer,
            a_buffer,
            b_buffer,
            ctx,
        )

    @staticmethod
    fn shape(
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
    ) -> IndexList[2]:
        var shape = a.shape()
        shape[1] = b.dim_size[1]()
        return rebind[IndexList[2]](shape)


@compiler.register("print_tensor_spec")
struct PrintTensorSpecOp:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        alias x_shape = compiler.specsof[x.type, x.rank]("x").shape
        alias x_strides = compiler.specsof[x.type, x.rank]("x").strides
        alias x_alignment = compiler.specsof[x.type, x.rank]("x").alignment
        alias x_address_space = compiler.specsof[x.type, x.rank](
            "x"
        ).address_space
        alias x_exclusive = compiler.specsof[x.type, x.rank]("x").exclusive

        print("x.shape =", x_shape)
        print("x.strides =", x_strides)
        print("x.alignment =", x_alignment)
        print("x.address_space =", x_address_space)
        print("x.exclusive =", x_exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x.load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("print_tensor_spec_view")
@compiler.view_kernel
struct PrintTensorSpecViewOp:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        alias x_shape = compiler.specsof[x.type, x.rank]("x").shape
        alias x_strides = compiler.specsof[x.type, x.rank]("x").strides
        alias x_alignment = compiler.specsof[x.type, x.rank]("x").alignment
        alias x_address_space = compiler.specsof[x.type, x.rank](
            "x"
        ).address_space
        alias x_exclusive = compiler.specsof[x.type, x.rank]("x").exclusive

        print("x.shape =", x_shape)
        print("x.strides =", x_strides)
        print("x.alignment =", x_alignment)
        print("x.address_space =", x_address_space)
        print("x.exclusive =", x_exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x.load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("print_tensor_spec_fused")
struct PrintTensorSpecFusedOp:
    @compiler.elementwise
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        alias x_shape = compiler.specsof[x.type, x.rank]("x").shape
        alias x_strides = compiler.specsof[x.type, x.rank]("x").strides
        alias x_alignment = compiler.specsof[x.type, x.rank]("x").alignment
        alias x_address_space = compiler.specsof[x.type, x.rank](
            "x"
        ).address_space
        alias x_exclusive = compiler.specsof[x.type, x.rank]("x").exclusive

        print("x.shape =", x_shape)
        print("x.strides =", x_strides)
        print("x.alignment =", x_alignment)
        print("x.address_space =", x_address_space)
        print("x.exclusive =", x_exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x._fused_load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("imposter_add_elementwise")
@compiler.elementwise
struct AddElementwise:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.type, width]](y._fused_load[width](idx))

        foreach[func](z)


@compiler.register("imposter_add_lhs")
struct AddFuseLHS:
    @compiler.enable_fusion_for("x")
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.type, width]](y.load[width](idx))

        # Wrapper to hide the foreach call from MOGGPreElab.
        # Otherwhise it would still be detected as an elementwise kernel.
        fn foo():
            foreach[func](z)

        foo()


@compiler.register("imposter_add_fuse_inputs")
struct AddFuseInputs:
    @compiler.enable_fusion_for("x", "y")
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.type, width]](y._fused_load[width](idx))

        # Wrapper to hide the foreach call from MOGGPreElab.
        # Otherwhise it would still be detected as an elementwise kernel.
        fn foo():
            foreach[func](z)

        foo()


# c = a @ b, should support CPU and GPU
@compiler.register("matmul_fuse_out")
struct MatmulFuseOut:
    @compiler.enable_fusion_for("c")
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        lambdas_have_fusion: Bool,
    ](
        c: ManagedTensorSlice,
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        alias rank = a.rank
        alias a_dtype = a.type
        alias b_dtype = b.type
        alias c_dtype = c.type

        # Convert everything to NDBuffer
        var c_buffer = toNDBuffer[c_dtype, 2](c)
        var a_buffer = toNDBuffer[a_dtype, 2](a)
        var b_buffer = toNDBuffer[b_dtype, 2](b)

        @parameter
        @always_inline
        fn out_func[
            type: DType, width: Int, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[type, width]):
            c._fused_store(idx, rebind[SIMD[c.type, width]](val))

        print("lambdas_have_fusion =", lambdas_have_fusion)

        _matmul[
            False,
            False,
            False,
            elementwise_lambda_fn=out_func,
            saturated_vnni=False,
            single_thread_blocking_override=synchronous,
            target=target,
        ](
            c_buffer,
            a_buffer,
            b_buffer,
            ctx,
        )

    @staticmethod
    fn shape(
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
    ) -> IndexList[2]:
        var shape = a.shape()
        shape[1] = b.dim_size[1]()
        return rebind[IndexList[2]](shape)


@compiler.register("op_with_synchronous")
struct WithSynchronous:
    @staticmethod
    fn execute[
        synchronous: Bool,
    ](out: ManagedTensorSlice, input: ManagedTensorSlice):
        print("what up ", synchronous)


@compiler.register("op_without_synchronous")
struct WithoutSynchronous:
    @staticmethod
    fn execute(out: ManagedTensorSlice, input: ManagedTensorSlice):
        print("what up")


# Simple, expects variadics to have the same size, and simply copies the first
# number from the associated inputs to outputs, plus a bias
@compiler.register("variadic_input_to_output")
struct VariadicInputToOutput:
    @staticmethod
    fn execute[
        type: DType,
        synchronous: Bool,
        size: Int,
        target: StringLiteral,
    ](
        output: StaticTuple[ManagedTensorSlice[type, rank=1], size],
        bias: ManagedTensorSlice[type, rank=1],
        input: StaticTuple[ManagedTensorSlice[type, rank=1], size],
    ):
        @parameter
        for i in range(size):
            for j in range(input[i].size()):
                output[i][j] = input[i][j]
            output[i][0] += bias[0]


# Simply adds the first number of bias to the first number of boath outputs
# Mainly here to test logic with multiple DPS outputs
@compiler.register("add_bias_to_two_tensors", num_dps_outputs=2)
struct AddBiasToDouble:
    @staticmethod
    fn execute[
        rank: Int,
        type: DType,
        synchronous: Bool,
    ](
        output1: ManagedTensorSlice[type, rank],
        output2: ManagedTensorSlice[type, rank],
        input1: ManagedTensorSlice[type, rank],
        input2: ManagedTensorSlice[type, rank],
        bias: ManagedTensorSlice[type, rank],
    ):
        output1[0] = input1[0] + bias[0]
        output2[0] = input2[0] + bias[0]


@compiler.register("inplace_increment_elem", num_dps_outputs=0)
struct BasicInplace:
    @compiler.mutable("input")
    @staticmethod
    fn execute[
        type: DType,
    ](input: ManagedTensorSlice[type, rank=2]):
        x = input[0, 0]
        x += 1
        input[0, 0] = x


# Have this nearly identical version as having a raise changes the Mojo function's signature
@compiler.register("inplace_increment_elem_raises", num_dps_outputs=0)
struct BasicInplaceRaises:
    @compiler.mutable("input")
    @staticmethod
    fn execute[
        type: DType,
    ](input: ManagedTensorSlice[type, rank=2]) raises:
        x = input[0, 0]
        x += 1
        input[0, 0] = x


@compiler.register("variadic_add")
struct VariadicAdd:
    @compiler.enable_fusion_for("inputs")
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, rank],
        inputs: StaticTuple[ManagedTensorSlice[type, rank], *_],
    ):
        alias inputs_specs = compiler.specsof[type, rank, inputs.size]("inputs")

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[type, width]:
            var acc = SIMD[type, width](0)

            @parameter
            for i in range(inputs.size):
                alias in_lambda = inputs_specs[i].in_lambda

                @parameter
                if in_lambda:
                    alias in_fn = in_lambda.value()
                    acc += in_fn[width](idx)
                else:
                    acc += inputs[i].load[width](idx)

            return acc

        # Wrapper to hide the foreach call from MOGGPreElab.
        # Otherwhise it would still be detected as an elementwise kernel.
        fn foo():
            foreach[func](output)

        foo()


@compiler.register("transpose_2d")
@compiler.view_kernel
struct Transpose2DOp:
    @staticmethod
    fn build_view[
        type: DType,
    ](x: ManagedTensorSlice[type, 2],) -> ManagedTensorSlice[type, 2]:
        var new_stride = IndexList[2]()
        var new_shape = IndexList[2]()
        new_stride[0] = x._runtime_strides[1]
        new_stride[1] = x._runtime_strides[0]
        new_shape[0] = x._spec.shape[1]
        new_shape[1] = x._spec.shape[0]

        return ManagedTensorSlice[type, 2](x._ptr, new_shape, new_stride)

    @staticmethod
    fn get_view_strides(input_strides: DimList) -> DimList:
        # transpose the strides of the input
        return DimList(input_strides.at[1](), input_strides.at[0]())

    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
    ](
        z: ManagedTensorSlice[type, 2],
        x: ManagedTensorSlice[type, 2],
        ctx: MojoCallContextPtr,
    ):
        alias x_strides = compiler.specsof[x.type, x.rank]("x").strides
        alias view_strides = Self.get_view_strides(x_strides)
        var x_view = Self.build_view(x)
        view_copy_impl[synchronous, target, view_strides=view_strides](
            z, x_view, ctx
        )


@compiler.register("elementwise_print_shape")
@compiler.elementwise
struct ElementwisePrintShape:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x._fused_load[width](idx))

        print("input.shape =", x._spec.shape)
        print("output.shape =", z._spec.shape)

        foreach[func](z)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


# Raises if input shape is 10
@compiler.register("custom_op_that_raises")
struct CustomOpThatRaises:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice) raises:
        if x.shape()[0] == 10:
            raise ("input_shape[0] == 10")

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x._fused_load[width](idx))

        foreach[func](z)

    @staticmethod
    fn shape(x: ManagedTensorSlice) raises -> IndexList[x.rank]:
        print("Hello")
        var out_shape = x.shape()
        if out_shape[0] == 20:
            raise ("data.get_shape()[0] == 20")
        return out_shape


@compiler.register("mo.test.failing_constraint")
struct OpThatAlwaysFailsConstraint:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        out_tensor: ManagedTensorSlice[type, rank],
        in_tensor: ManagedTensorSlice[type, rank],
    ):
        constrained[
            1 == 2,
            "Expected constraint failure for error message testing",
        ]()


@compiler.register("mo.test.return_error")
struct OpThatAlwaysRaises:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        out_tensor: ManagedTensorSlice[type, rank],
        in_tensor: ManagedTensorSlice[type, rank],
    ) raises:
        out_tensor[0] = in_tensor[0]
        raise Error("This is an error")


@compiler.register("monnx.abs_v13")
struct MONNXAbsOverload:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        out_tensor: ManagedTensorSlice[type, rank],
        in_tensor: ManagedTensorSlice[type, rank],
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[type, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor)


@compiler.register("torch.aten.abs")
struct MTorchAbsOverload:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        out_tensor: ManagedTensorSlice[type, rank],
        in_tensor: ManagedTensorSlice[type, rank],
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[type, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor)


@compiler.register("op_with_custom_params")
struct OpWithCustomParams:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        custom_int: Int,
        custom_str: StringLiteral,
        custom_dtype: DType,
    ](
        out_tensor: ManagedTensorSlice[type, rank],
        in_tensor: ManagedTensorSlice[type, rank],
    ) raises:
        out_tensor[0] = in_tensor[0]
        print("custom_int =", custom_int)
        print("custom_str =", custom_str)
        print("custom_dtype =", custom_dtype)


@compiler.register("mgprt_test_func")
struct MGPRTTestFunc:
    @staticmethod
    fn execute(out_tensor: ManagedTensorSlice) raises:
        external_call["MGP_RT_TEST", NoneType]()


@compiler.register("mutable_test_op", num_dps_outputs=0)
struct MutableTestOp:
    @staticmethod
    fn execute(in_place_tensor: ManagedTensorSlice) raises:
        in_place_tensor._ptr.store(0, 0)


# For testing support for Scalar[...] in Mojo
@compiler.register("supports_scalar_kernel")
struct SupportsScalarKernel:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type, 1],
        x: ManagedTensorSlice[type, 1],
        y: Scalar[type],
    ) raises:
        print("datatype is", type)


@compiler.register("kernel_with_no_target")
struct KernelWithNoTarget:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type, *_], x: ManagedTensorSlice[type, *_]
    ) raises:
        print("hello from kernel with no target")


@compiler.register("basic_target")
struct BasicTarget:
    @staticmethod
    fn execute[
        type: DType, target: StringLiteral
    ](
        out: ManagedTensorSlice[type, *_], x: ManagedTensorSlice[type, *_]
    ) raises:
        print("hello from kernel on", target)


@value
@register_passable
struct MyCustomScalarReg[type: DType]:
    var val: Scalar[type]

    @implicit
    fn __init__(out self, val: Scalar[type]):
        print("MyCustomScalarReg.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarReg.__del__", self.val)


@compiler.register("buff_to_my_custom_scalar_reg", num_dps_outputs=0)
struct BuffToMyCustomScalarReg:
    @uses_opaque
    @staticmethod
    fn execute[
        target: StringLiteral
    ](x: ManagedTensorSlice[DType.int32, 1]) -> MyCustomScalarReg[DType.int32]:
        return MyCustomScalarReg(x[0])


@compiler.register("my_custom_scalar_reg_to_buff")
struct CustomScalarRegToBuff:
    @uses_opaque
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        input: ManagedTensorSlice[DType.int32, 1],
        x: MyCustomScalarReg[DType.int32],
    ):
        input[0] = x.val


@compiler.register("test_custom_op")
struct TestCustomOp:
    @staticmethod
    fn execute[
        target: StringLiteral, type: DType, rank: Int
    ](
        out: ManagedTensorSlice[type, rank],
        input: ManagedTensorSlice[type, rank],
    ):
        print("World!")

    @staticmethod
    fn shape[
        type: DType, rank: Int
    ](input: ManagedTensorSlice[type, rank]) -> IndexList[rank]:
        print("Hello")
        return input.shape()


@compiler.register("invalid_kernel_owned_arg", num_dps_outputs=0)
struct InvalidOwnedArgConvention:
    @staticmethod
    fn execute[
        target: StringLiteral, type: DType, rank: Int
    ](owned input: MyCustomScalarSI32) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(input.val)


@compiler.register("single_device_context")
struct SingleDeviceContext:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type, *_],
        x: ManagedTensorSlice[type, *_],
        dev_ctx: DeviceContextPtr,
    ) raises:
        dev_ctx[].synchronize()


@compiler.register("multi_device_context", num_dps_outputs=1)
struct MultiDeviceContext:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type, *_],
        x: ManagedTensorSlice[type, *_],
        dev_ctx0: DeviceContextPtr,
        dev_ctx1: DeviceContextPtr,
        ctx: MojoCallContextPtr,
    ) raises:
        print("dev_ctx0.id() =", dev_ctx0[].id())
        print("dev_ctx1.id() =", dev_ctx1[].id())
        dev_ctx0[].synchronize()
        dev_ctx1[].synchronize()


@compiler.register("multi_device_context_dedup")
struct MultiDeviceContextDedup:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type, *_],
        x: ManagedTensorSlice[type, *_],
        y: ManagedTensorSlice[type, *_],
        dev_ctx0: DeviceContextPtr,
        dev_ctx1: DeviceContextPtr,
    ) raises:
        dev_ctx0[].synchronize()
        dev_ctx1[].synchronize()
