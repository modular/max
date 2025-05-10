# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Linear algebra operations."""

from math import max

from max.graph.type import Dim, ElementType, MOTensor
from max.graph.type_promotion import promote
from max.graph.ops.casting import reshape


def outer(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the outer product of two symbolic vectors.

    Args:
        lhs: The left side of the product. Whatever its shape,
            it will be flattened to a rank-1 vector.
        rhs: The right side of the product. Whatever its shape,
            it will be flattened to a rank-1 vector. Must have the
            same number of elements as `lhs`.

    Returns:
        A symbolic tensor representing the
        [outer product](https://en.wikipedia.org/wiki/Outer_product)
        of the two input vectors. It will have rank 2, with the dimension
        sizes being the number of elements of `lhs` and `rhs` respectively.
    """
    return lhs.reshape(-1, 1) * rhs.reshape(1, -1)


def matmul_broadcast(lhs: Symbol, rhs: Symbol) -> SymbolTuple:
    """Computes the broadcasting of two symbolic tensors for a matmul.

    Args:
        lhs: The left side of the matmul.
        rhs: The right side of the matmul.

    Returns:
        A pair of symbolic tensors corresponding to the `lhs` and `rhs`
        respectively, after being broadcast to the right shapes to perform
        a matmul between them. This is similar to an `elementwise_broadcast`
        except in the final two dimensions of each tensor. The last dimension
        of `lhs` is broadcast against the 2nd-to-last dimension of `rhs`, while
        the 2nd-to-last dimension of `lhs` and the last dimension of `rhs` are
        untouched.
    """
    var g = lhs.graph()
    var lhs_type = lhs.tensor_type()
    var rhs_type = rhs.tensor_type()

    var lhs_rank = lhs_type.rank()
    var rhs_rank = rhs_type.rank()

    var broadcast_rank = max(lhs_rank, rhs_rank)
    var lhs_shape = shape_of(lhs)
    var rhs_shape = shape_of(rhs)

    var lhs_broadcast_dims = lhs_shape[: lhs_rank - 2]
    var lhs_matrix_dims = lhs_shape[lhs_rank - 2 : lhs_rank]

    var rhs_broadcast_dims = rhs_shape[: rhs_rank - 2]
    var rhs_matrix_dims = rhs_shape[rhs_rank - 2 : rhs_rank]

    var broadcast_dims_shape = g.op(
        "mo.broadcast_shape",
        (lhs_broadcast_dims, rhs_broadcast_dims),
        MOTensor(DType.int64, broadcast_rank - 2),
    )

    var lhs_final_dims = List[Dim]()
    var rhs_final_dims = List[Dim]()
    for _ in range(broadcast_rank - 2):
        lhs_final_dims.push_back(Dim.dynamic())
        rhs_final_dims.push_back(Dim.dynamic())
    lhs_final_dims.push_back(lhs_type.dim(-2))
    lhs_final_dims.push_back(lhs_type.dim(-1))
    rhs_final_dims.push_back(rhs_type.dim(-2))
    rhs_final_dims.push_back(rhs_type.dim(-1))

    var lhs_broadcast_shape = concat((broadcast_dims_shape, lhs_matrix_dims))

    var broadcast_lhs = g.op(
        "mo.broadcast_to",
        (lhs, lhs_broadcast_shape),
        MOTensor(lhs_type.dtype, lhs_final_dims),
    )

    var rhs_broadcast_shape = concat((broadcast_dims_shape, rhs_matrix_dims))

    var broadcast_rhs = g.op(
        "mo.broadcast_to",
        (rhs, rhs_broadcast_shape),
        MOTensor(rhs_type.dtype, rhs_final_dims),
    )

    return (broadcast_lhs, broadcast_rhs)


def matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the matrix multiplication of two symbolic tensors.

    The last two dimensions of each tensor are treated as matricies and multiplied,
    and the remaining dimensions are broadcast dimensions.

    Args:
        lhs: The left-hand-side of the matmul.
        rhs: The right-hand-side of the matmul.

    Returns:
        A symbolic tensor representing he result of broadcasting the two
        matricies together according to `matmul_broadcast` and then performing
        a matrix multiply along the last two dimension of each tensor.
    """
    var rhs_type = rhs.tensor_type()
    if rhs_type.rank() > 2:
        return batch_matmul(lhs, rhs)
    else:
        return matmul_by_matrix(lhs, rhs)


def batch_matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the matrix multiplication of two symbolic tensors.

    The last two dimensions of each tensor are treated as matricies and multiplied,
    and the remaining dimensions are broadcast dimensions.

    This supports arbitrary-rank `rhs` inputs, but may be less performant than
    `matmul_by_matrix`.

    Args:
        lhs: The left-hand-side of the matmul.
        rhs: The right-hand-side of the matmul.

    Returns:
        A symbolic tensor representing he result of broadcasting the two
        matricies together according to `matmul_broadcast` and then performing
        a matrix multiply along the last two dimension of each tensor.
    """
    var g = lhs.graph()
    var broadcast_pair = matmul_broadcast(lhs, rhs)
    var broadcast_lhs = broadcast_pair[0]
    var broadcast_rhs = broadcast_pair[1]

    var lhs_type = broadcast_lhs.tensor_type()
    var rhs_type = broadcast_rhs.tensor_type()
    var dims = List[Dim]()
    for i in range(lhs_type.rank() - 1):
        dims.push_back(lhs_type.dims[i])
    dims.push_back(rhs_type.dim(-1))
    var out_type = MOTensor(lhs_type.dtype, dims)

    return g.op("mo.batch_matmul", (broadcast_lhs, broadcast_rhs), out_type)


def matmul_by_matrix(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Computes the matrix multiplication of two symbolic tensors.

    The last two dimensions of each tensor are treated as matricies and multiplied,
    and the remaining dimensions are broadcast dimensions.

    Args:
        lhs: The left-hand-side of the matmul.
        rhs: The right-hand-side of the matmul. Must have rank exactly 2.

    Returns:
        A symbolic tensor representing he result of broadcasting the two
        matricies together according to `matmul_broadcast` and then performing
        a matrix multiply along the last two dimension of each tensor.
    """
    var g = lhs.graph()
    var lhs_type = lhs.tensor_type()
    var rhs_type = rhs.tensor_type()
    if rhs_type.rank() != 2:
        raise "rhs must be a matrix"

    var lhs_shape = shape_of(lhs)
    var rhs_shape = shape_of(rhs)
    last_lhs_axis = lhs_type.rank() - 1
    var reshape_shape = stack((g.scalar(Int64(-1)), lhs_shape[last_lhs_axis]))
    var final_shape = concat((lhs_shape[:last_lhs_axis], rhs_shape[1:2]))

    var final_dims = List[Dim]()
    for i in range(lhs_type.rank() - 1):
        final_dims.push_back(lhs_type.dim(i))
    final_dims.push_back(rhs_type.dim(-1))

    var matmul_dims = List[Dim]()
    matmul_dims.append(Dim.dynamic())
    matmul_dims.append(lhs_type.dim(-1))
    var matmul_out = g.op(
        "mo.matmul",
        (reshape(lhs, reshape_shape, matmul_dims), rhs),
        MOTensor(lhs_type.dtype, Dim.dynamic(), rhs_type.dim(-1)),
    )

    return reshape(matmul_out, final_shape, final_dims)


def band_part(
    input: Symbol, num_lower: Symbol, num_upper: Symbol, exclude: Bool = False
) -> Symbol:
    """Masks out everything except a diagonal band of an input matrix.

    Copies a tensor setting everything outside the central diagonal band of the
    matricies to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matricies.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values are given by

    ```
    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].
    ```

    with the indicator function:

    ```
    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))
    ```

    Args:
        input: The input to mask out.
        num_lower: The number of diagonal bands to include below the central
            diagonal. If -1, include the entire lower triangle.
        num_upper: The number of diagonal bands to include above the central
            diagonal. If -1, include the entire upper triangle.
        exclude: If true, invert the selection of elements to mask. Elements
            in the band are set to zero.
    Returns:
        A symbolic tensor value with the configured selection masked out
        to 0 values, and the remaining values copied from the input tensor.
    """
    var g = input.graph()
    return g.op(
        "mo.linalg.band_part",
        (
            input,
            num_lower.reshape(),
            num_upper.reshape(),
            g.scalar[DType.bool](exclude),
        ),
        input.type(),
    )
