# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm import vectorize
from algorithm.functional import _elementwise_impl
from memory.buffer import NDBuffer
from runtime.llcl import MojoCallContextPtr

from utils._annotations import *
from utils.index import StaticIntTuple
from utils.list import DimList

# ===----------------------------------------------------------------------===#
# Special test targets just for generation tests
# ===----------------------------------------------------------------------===#


@mogg_register("test_many_ranks_and_types")
@export
fn test_many_ranks_and_types[
    type1: DType,
    rank1: Int,
    type2: DType,
    rank2: Int,
    type3: DType,
    rank3: Int,
    type4: DType,
    rank4: Int,
    type5: DType,
    rank5: Int,
](
    tensor1: NDBuffer[type1, rank1],
    tensor2: NDBuffer[type2, rank2],
    tensor3: NDBuffer[type3, rank3],
    tensor4: NDBuffer[type4, rank4],
    tensor5: NDBuffer[type5, rank5],
) -> NDBuffer[type1, rank1]:
    """
    Used as a test target to ensure parameter deduction works when there are
    many to deduce and also used to check errors.
    """
    return tensor1


@mogg_register("test_one_rank_many_tensor")
@export
fn test_one_rank_many_tensor[
    type: DType, rank: Int
](
    tensor1: NDBuffer[type, rank],
    tensor2: NDBuffer[type, rank],
    tensor3: NDBuffer[type, rank],
    tensor4: NDBuffer[type, rank],
    tensor5: NDBuffer[type, rank],
) -> NDBuffer[type, rank]:
    """
    Used as a test target to ensure we can deduce type and rank when used by
    many arguments.
    """
    return tensor1


@mogg_register("test_3D_in_out_lambda")
@export
fn test_3D_in_out_lambda[
    type: DType,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](tensor1: NDBuffer[type, 3], output: NDBuffer[type, 3],) -> NDBuffer[type, 3]:
    """
    Used as a target to test passing input and output lambdas.
    """

    for x in range(0, tensor1.dim[0]()):
        for y in range(0, tensor1.dim[1]()):

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                let indices = StaticIntTuple[3](x, y, idx)
                let result = input_0_fn[simd_width, 3](indices)
                output_0_fn[simd_width, 3](indices, result)

            vectorize[
                func_wrapper,
                simd_width,
            ](tensor1.dim[2]())

    return output


@mogg_register("test_indices_deduction")
@export
fn _test_indices_deduction[
    num_indices: Int
](indices: StaticIntTuple[num_indices]):
    """
    Used as a test to make sure we correctly deduce the size of indices.
    """
    print("Indices size: ")
    print(num_indices)
    print("Indices: ")
    print(indices)


@mogg_register("test_make_indices")
@export
fn _test_make_indices[num_indices: Int]() -> StaticIntTuple[num_indices]:
    """
    Used to return indices which we can use as a target for tests.
    """
    var out = StaticIntTuple[num_indices]()
    for i in range(num_indices):
        out[i] = i
    return out


@mogg_register_override("mo.sqrt", 1)
@mogg_elementwise
@export
fn sqrt_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    print("In override sqrt")
    return value


@mogg_register("test_static_shape_deduction")
@export
fn test_static_shape_deduction[
    type: DType, rank: Int, input_0_static_shape: DimList
](tensor: NDBuffer[type, rank, input_0_static_shape],):
    print("Printing shape: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = input_0_static_shape.at[idx]()

        @parameter
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[rank, body]()


@mogg_register("test_static_shape_output")
@export
fn test_static_shape_output[
    type: DType, rank: Int, output_0_static_shape: DimList
]() -> NDBuffer[type, rank, output_0_static_shape]:
    print("Printing output shape: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = output_0_static_shape.at[idx]()
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[rank, body]()
    return NDBuffer[type, rank, output_0_static_shape](
        DTypePointer[type](), StaticIntTuple[rank](), StaticIntTuple[rank]()
    )


@mogg_register("test_int_list_param")
@export
fn test_int_list_param[length: Int, int_list: DimList]():
    print("Printing parameter: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = int_list.at[idx]()
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[length, body]()


@mogg_register("test_custom_op")
@always_inline
@export
fn test_unary_kernel[
    type: DType,
    rank: Int,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    single_thread_blocking_override: Bool,
](input_shape: StaticIntTuple[rank], output_shape: StaticIntTuple[rank],):
    print("World!")


@mogg_register_shape_func("test_custom_op")
@always_inline
@export
fn test_unary_kernel_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank],) -> StaticIntTuple[rank]:
    print("Hello")

    return data.get_shape()


@mogg_register("test_custom_op_params")
@always_inline
@export
fn test_unary_kernel_params[
    type: DType,
    rank: Int,
    extra_param: Int,
    extra_param2: StringLiteral,
](tensor1: NDBuffer[type, rank], output: NDBuffer[type, rank],):
    print(extra_param)
    print(extra_param2)


@mogg_register("tf.Identity")
@mogg_register("torch.aten.abs")
@mogg_register("monnx.abs_v13")
@always_inline
@export
fn test_custom_identity[
    type: DType,
    rank: Int,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    single_thread_blocking_override: Bool,
](input_shape: StaticIntTuple[rank], output_shape: StaticIntTuple[rank],):
    print("The custom identity op is running!")

    @parameter
    @always_inline
    fn identity[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        let x = input_0_fn[simd_width, rank](idx)
        output_0_fn[simd_width, rank](idx, x)

    _elementwise_impl[
        rank,
        simd_width,
        single_thread_blocking_override,
        identity,
        target="cpu",
    ](
        input_shape,
    )


@mogg_register_shape_func("tf.Identity")
@mogg_register_shape_func("torch.aten.abs")
@mogg_register_shape_func("monnx.abs_v13")
@always_inline
@export
fn test_custom_identity_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank],) -> StaticIntTuple[rank]:
    return data.get_shape()


@mogg_register("test_variadic")
@always_inline
@export
fn concat(
    ctx: MojoCallContextPtr,
    *variadic_ins: NDBuffer[DType.float32, 1],
):
    pass


@mogg_register_shape_func("test_custom_op_inline")
@export
fn reduce_shape_no_explicit_inline[
    input_rank: Int,
    input_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    axis_buf: NDBuffer[axis_type, 1],
) -> StaticIntTuple[input_rank]:
    # extract hyper parameter
    var axis = int(axis_buf[0])
    if axis < 0:
        axis += input_rank

    # compute and return the output shape
    var output_shape = input_buf.get_shape()
    output_shape[axis] = 1
    return output_shape


@mogg_register("custom_op_that_raises")
@export
fn custom_op_that_raises[
    type: DType,
    rank: Int,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    single_thread_blocking_override: Bool,
](input_shape: StaticIntTuple[rank], output_shape: StaticIntTuple[rank]) raises:
    if input_shape[0] == 10:
        raise ("input_shape[0] == 10")

    @parameter
    @always_inline
    fn identity[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        let x = input_0_fn[simd_width, rank](idx)
        output_0_fn[simd_width, rank](idx, x)

    _elementwise_impl[
        rank,
        simd_width,
        single_thread_blocking_override,
        identity,
        target="cpu",
    ](
        input_shape,
    )


@mogg_register_shape_func("custom_op_that_raises")
@always_inline
@export
fn custom_shape_func_that_raises[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank],) raises -> StaticIntTuple[rank]:
    # This print ensures we won't symbolicize this shape function call, so we
    # can test its runtime execution.
    print("Hello")

    let out_shape = data.get_shape()
    if out_shape[0] == 20:
        raise ("data.get_shape()[0] == 20")
    return out_shape
