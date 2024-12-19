# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Token sampling algorithms."""

from typing import Optional

from max.dtype import DType
from max.graph import Dim, Graph, Shape, TensorType, TensorValue, ops


def token_sampler(top_k: Optional[int], in_dtype: DType, out_dtype: DType):
    logits_in_type = TensorType(in_dtype, ["batch", "vocab_size"])
    prev_tokens_type = TensorType(DType.int64, ["batch", "num_prev_steps"])
    with Graph(
        "token_sampler", input_types=[logits_in_type, prev_tokens_type]
    ) as graph:
        logits, prev_tokens = (val.tensor for val in graph.inputs)
        logits = ops.cast(logits, out_dtype)
        if top_k is not None:
            shape = Shape(logits.shape)
            shape[-1] = Dim(1)
            tokens = ops.custom(
                "topk_fused_sampling",
                [ops.constant(top_k, dtype=DType.int64), logits],
                [TensorType(DType.int64, shape)],
            )[0]
            assert isinstance(tokens, TensorValue)
        else:
            tokens = ops.argmax(logits)

        all_tokens = ops.concat([prev_tokens, tokens], -1)
        tokens = ops.squeeze(tokens, -1)
        graph.output(tokens, all_tokens)

        return graph


def argmax_sampler(dtype: DType):
    logits_type = TensorType(dtype, ["batch", "vocab_size"])
    return Graph("argmax", ops.argmax, input_types=[logits_type])
