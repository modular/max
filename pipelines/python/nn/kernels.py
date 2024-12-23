# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Helper functions for wrapping custom kv cache/attention related ops."""

from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np
from max.dtype import DType
from max.graph import Dim, TensorType, TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheCollection,
)


def fused_qkv_ragged_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: Union[
        ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
    ],
    layer_idx: TensorValue,
    n_heads: int,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input.

    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.dtype != wqkv.dtype:
        msg = (
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )
        raise ValueError(msg)

    input_rank_expected = 2
    if input.rank != input_rank_expected:
        msg = f"expected input to have rank {input_rank_expected}, was {input.rank}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        cache_strategy_str = "cont_batch"
    elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
        cache_strategy_str = "paged"
    else:
        msg = f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    op_name = f"fused_qkv_matmul_kv_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_{cache_strategy_str}_ragged"

    return ops.inplace_custom(
        op_name,
        values=[input, input_row_offsets, wqkv, kv_collection, layer_idx],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
    )[0].tensor


def fused_qkv_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    wqkv: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    n_heads: int,
) -> TensorValue:
    """Computes fused query, key and value projections."""
    if input.dtype != wqkv.dtype:
        msg = (
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )
        raise ValueError(msg)

    input_rank_expected = 3
    if input.rank != input_rank_expected:
        msg = f"expected input to have rank {input_rank_expected}, was {input.rank}"
        raise ValueError(msg)

    wqkv_rank_expected = 2
    if wqkv.rank != wqkv_rank_expected:
        msg = (
            f"expected wqkv to have rank {wqkv_rank_expected}, was {wqkv.rank}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for fused_qkv_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    op_name = f"fused_qkv_matmul_kv_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_bshd_continuous_batch"

    return ops.inplace_custom(
        op_name,
        values=[input, wqkv, kv_collection, layer_idx],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
    )[0].tensor


def matmul_kv_cache_ragged(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: int | np.integer,
) -> None:
    """Computes key and value projections with ragged input.

    `hidden_states` and `input_row_offsets` are used together to
    implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """
    if hidden_states.dtype != weight.dtype:
        msg = (
            "expected hidden_states and weight to have the same dtype, but got"
            f" {hidden_states.dtype} and {weight.dtype}, respectively."
        )
        raise ValueError(msg)

    hidden_states_rank_expected = 2
    if hidden_states.rank != hidden_states_rank_expected:
        msg = (
            "expected hidden_states to have rank "
            f"{hidden_states_rank_expected}, was {hidden_states.rank}"
        )
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for fused_qkv_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    ops.inplace_custom(
        name=f"matmul_kv_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_cont_batch_ragged",
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection,
            ops.constant(layer_idx, DType.uint32),
        ],
    )


def fused_qk_ragged_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: TensorValue,
    layer_idx: TensorValue,
    interleaved: bool = True,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings and ragged inputs.

    `input` and `input_row_offsets` are used together to implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """

    if input.dtype != freqs_cis.dtype:
        msg = (
            "expected input and freqs_cis to share a dtype, but got"
            f" {input.dtype} and {freqs_cis.dtype} respectively"
        )
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        cache_strategy_str = "continuous_batch"
    elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
        cache_strategy_str = "paged"
    else:
        msg = f"unsupported cache strategy for fused_qk_ragged_rope: {kv_params.cache_strategy}"
        raise ValueError(msg)

    op_name = f"fused_qk_rope_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_bshd_{cache_strategy_str}_ragged"

    return ops.inplace_custom(
        op_name,
        values=[
            input,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            ops.constant(interleaved, DType.bool),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def fused_qk_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis_2d: TensorValue,
    layer_idx: TensorValue,
    interleaved: bool = True,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings."""
    input_rank_expected = 4
    if input.rank != input_rank_expected:
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    freqs_cis_rank_expected = 2
    if freqs_cis_2d.rank != freqs_cis_rank_expected:
        msg = (
            f"expected freqs_cis_2d of rank {freqs_cis_rank_expected} but got "
            f"{freqs_cis_2d.rank}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for fused_qkv_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    op_name = f"fused_qk_rope_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_bshd_continuous_batch"

    return ops.inplace_custom(
        op_name,
        values=[
            input,
            kv_collection,
            freqs_cis_2d,
            layer_idx,
            ops.constant(interleaved, DType.bool),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def flash_attention(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    attention_mask: TensorValue,
    valid_lengths: TensorValue,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache."""
    input_rank_expected = 4
    if input.rank != input_rank_expected:
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if attention_mask.dtype != input.dtype:
        msg = (
            f"expected attention mask dtype {attention_mask.dtype} to match "
            f"the input's dtype {input.dtype}"
        )
        raise ValueError(msg)

    if valid_lengths.dtype != DType.uint32:
        msg = f"expected uint32 valid_lengths but got {valid_lengths.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for flash_attention: {kv_params.cache_strategy}"
        raise ValueError(msg)

    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_bshd_continuous_batch"

    # NOTE: The scale argument to the flash attention kernel is constrained to
    # float32.
    scale = ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.float32))
    return ops.inplace_custom(
        op_name,
        values=[
            input,
            kv_collection,
            layer_idx,
            attention_mask,
            valid_lengths,
            scale,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def flash_attention_with_causal_mask(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache.
    Notably, materializes the causal mask within the kernel."""

    if input.shape[0] != valid_lengths.shape[0]:
        msg = (
            "expected batch size of input, to equal length of valid_lengths"
            f" got batch size of input ({input.shape[0]}), length of"
            f" valid_lengths ({valid_lengths.shape[0]})"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if valid_lengths.dtype != DType.uint32:
        msg = f"expected uint32 valid_lengths but got {valid_lengths.dtype}"
        raise ValueError(msg)

    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_causal_mask_continuous_batch"
    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for flash_attention: {kv_params.cache_strategy}"
        raise ValueError(msg)

    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_causal_mask_continuous_batch"

    # NOTE: The scale argument to flash attention is constrained to float32.
    scale = ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.float32))
    return ops.inplace_custom(
        op_name,
        values=[input, kv_collection, layer_idx, valid_lengths, scale],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


class MaskVariant(str, Enum):
    CAUSAL_MASK = "causal_mask"
    ALIBI_MASK = "alibi_mask"
    NULL_MASK = "null_mask"

    def __str__(self) -> str:
        return self.value


def flash_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    mask_variant: MaskVariant,
) -> TensorValue:
    """Computes flash (self) attention provided the `!mo.opaque` KV Cache.

    Notably, this materializes the attention mask (dependent on MaskVariant)
    within the kernel.
    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    Note that this is self attention and the KV sequence length is
    assumed to be equal to the Q sequence length.
    For KV sequence length != Q sequence length, use `cross_attention_ragged`.
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        cache_strategy_str = "cont_batch"
    elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
        cache_strategy_str = "paged"
    else:
        msg = f"unsupported cache strategy for flash_attention_ragged: {kv_params.cache_strategy}"
        raise ValueError(msg)

    op_name = f"flash_attention_kv_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_{str(mask_variant)}_{cache_strategy_str}_ragged"

    # NOTE: The scale argument to flash attention is constrained to float32.
    scale = ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.float32))

    return ops.inplace_custom(
        op_name,
        values=[input, input_row_offsets, kv_collection, layer_idx, scale],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def cross_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    mask_variant: MaskVariant,
    kv_input_row_offsets: TensorValue,
    q_max_seq_len: TensorValue,
) -> TensorValue:
    """Computes cross attention provided the `!mo.opaque` KV Cache.

    Notably, this materializes the attention mask (dependent on MaskVariant)
    within the kernel.
    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    attention, `kv_input_row_offsets` represents the KV sequence length.
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        cache_strategy_str = "cont_batch"
    elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
        cache_strategy_str = "paged"
    else:
        msg = f"unsupported cache strategy for flash_attention_ragged: {kv_params.cache_strategy}"
        raise ValueError(msg)

    if q_max_seq_len and (q_max_seq_len.dtype != DType.uint32):
        msg = (
            "expected q_max_seq_len to be uint32 but got {q_max_seq_len.dtype}"
        )
        raise ValueError(msg)

    op_name = f"cross_attention_kv_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_{str(mask_variant)}_{cache_strategy_str}_ragged"

    # NOTE: The scale argument to flash attention is constrained to float32.
    scale = ops.rsqrt(ops.constant(kv_params.head_dim, dtype=DType.float32))

    return ops.inplace_custom(
        op_name,
        values=[
            input,
            input_row_offsets,
            # Plumb in the query max sequence length for cross attention.
            # For self attention this is the same as the KV max seq len stored
            # on the kv_collection, but that isn't the case for cross attention.
            q_max_seq_len,
            kv_input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
    )[0].tensor


def swish_glu(
    a: TensorValueLike, b0: TensorValueLike, b1: TensorValueLike
) -> TensorValue:
    """Computes swish(a@b0.t()) * (a@b1.t())"""
    a = TensorValue(a)
    b0 = TensorValue(b0)
    b1 = TensorValue(b1)
    a_rank_expected = 2
    if a.rank != a_rank_expected:
        msg = f"expected a to have rank {a_rank_expected}, was {a.rank}"
        raise ValueError(msg)

    b0_rank_expected = 2
    if b0.rank != b0_rank_expected:
        msg = f"expected b0 to have rank {b0_rank_expected}, was {b0.rank}"
        raise ValueError(msg)

    b1_rank_expected = 2
    if b1.rank != b1_rank_expected:
        msg = f"expected b1 to have rank {b1_rank_expected}, was {b1.rank}"
        raise ValueError(msg)

    m = a.shape[0]
    n = b0.shape[0]
    if b0.shape[1] != a.shape[1]:
        msg = f"a.shape[1] == {a.shape[1]} != {b0.shape[1]} == b0.shape[1]"
        raise ValueError(msg)

    if b0.shape != b1.shape:
        msg = f"b0.shape == {b0.shape} != {b1.shape} == b1.shape"
        raise ValueError(msg)

    if a.dtype != b0.dtype or a.dtype != b1.dtype:
        msg = (
            "Element types of all arguments must be equal, but received"
            f" {a.dtype}, {b0.dtype}, and {b1.dtype}."
        )
        raise ValueError(msg)

    return ops.custom(
        "swishGLU",
        values=[a, b0, b1],
        out_types=[
            TensorType(
                dtype=a.dtype,
                shape=[m, n],
                device=a.device,
            )
        ],
    )[0].tensor


def rms_norm_key_cache(
    kv_params: KVCacheParams,
    kv_collection: ContinuousBatchingKVCacheCollection,
    gamma: TensorValue,
    epsilon: float | np.floating,
    layer_idx: int | np.integer,
    total_seq_len: Dim,
    input_row_offsets: TensorValue,
) -> None:
    """Computes RMSNorm on the _new_ entries in the KVCache.

    Currently, the KVCacheT class itself isn't aware of the new cache entries
    until cache length increment, which happens after model forward.
    So use `input_row_offsets` to do this bookkeeping.
    """
    op_name = f"rms_norm_key_cache_h{kv_params.n_kv_heads_per_device}_d{kv_params.head_dim}_cont_batch"

    gamma_rank_expected = 1
    if gamma.rank != gamma_rank_expected:
        msg = (
            f"expected gamma of rank {gamma_rank_expected} but got {gamma.rank}"
        )
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    ops.inplace_custom(
        op_name,
        values=[
            kv_collection,
            gamma,
            ops.constant(epsilon, gamma.dtype),
            ops.constant(layer_idx, DType.uint32),
            ops.cast(TensorValue.from_dim(total_seq_len), DType.uint32),
            input_row_offsets,
        ],
    )
