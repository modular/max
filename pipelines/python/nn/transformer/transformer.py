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

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)

from ..attention.interfaces import AttentionImpl, AttentionImplQKV
from ..embedding import Embedding
from ..layer import Layer
from ..linear import MLP, Linear
from ..norm import LPLayerNorm, RMSNorm
from ..sequential import Sequential


@dataclass
class TransformerBlock(Layer):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    attention: AttentionImpl | AttentionImplQKV
    mlp: MLP | Sequential
    attention_norm: RMSNorm | LPLayerNorm
    mlp_norm: RMSNorm | LPLayerNorm

    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue:
        attn_out = self.attention(
            self.attention_norm(x),
            kv_collection,
            **kwargs,
        )

        h = x + attn_out
        h = h + self.mlp(self.mlp_norm(h))

        return h


@dataclass
class Transformer(Layer):
    """Transformer model consisting for TransformerBlock layers."""

    dim: int
    n_heads: int
    layers: list[TransformerBlock]
    norm: RMSNorm | LPLayerNorm
    output: Linear
    embedding: Embedding
    kv_params: KVCacheParams
    kv_collection_constructor: (
        FetchContinuousBatchingKVCacheCollection | FetchPagedKVCacheCollection
    )
    all_logits: bool = False

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        # TODO: Split into a ragged and non-ragged version.
        h = self.embedding(tokens)

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        for _, layer in enumerate(self.layers):
            h = layer(h, kv_collection, **kwargs)

        normalized = self.norm(h)

        if "input_row_offsets" in kwargs:
            # Ragged inputs/activations
            last_indices = kwargs["input_row_offsets"][1:] - 1
            last_tokens = ops.gather(normalized, last_indices, axis=0)
        else:
            # Dense padded inputs/activations
            valid_lengths = kwargs["valid_lengths"]
            # TODO: Remove once `gather_nd` works with nonstatic last dims.
            indices = ops.unsqueeze(valid_lengths - 1, -1)
            last_tokens = ops.gather_nd(normalized, indices, batch_dims=1)

        # Always return float32 logits, no matter the activation type.
        last_token_logits = ops.cast(self.output(last_tokens), DType.float32)

        if self.all_logits:
            all_logits = ops.cast(self.output(normalized), DType.float32)
            return (last_token_logits, all_logits)

        return (last_token_logits,)
