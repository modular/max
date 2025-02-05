# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""An opaque KV Cache optimized attention mechanism with Rope."""

from dataclasses import dataclass
from typing import List, Optional, Union

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    PagedKVCacheCollection,
)

from ..kernels import (
    MHAMaskVariant,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)
from ..rotary_embedding import OptimizedRotaryEmbedding
from .interfaces import (
    AttentionImpl,
    AttentionImplQKV,
    DistributedAttentionImpl,
)


@dataclass
class AttentionWithRope(AttentionImpl):
    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow

    rope: OptimizedRotaryEmbedding
    bias: Optional[TensorValue] = None

    def __call__(
        self,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            n_heads=self.n_heads,
            bias=self.bias,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if xq.device is not None:
            freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            self.layer_idx,
            interleaved=self.rope.interleaved,
        )

        # Calculate Flash Attention.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=self.layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.wo(attn_out)


def distribute_value(
    v: TensorValue, devices: List[DeviceRef]
) -> List[TensorValue]:
    return [v.to(device) for device in devices]


@dataclass
class DistributedAttentionWithRope(DistributedAttentionImpl):
    list_of_attentions: List[AttentionWithRope]
    devices: list[DeviceRef]

    def __call__(
        self,
        x: List[TensorValue],
        kv_collections: List[ContinuousBatchingKVCacheCollection],
        **kwargs,
    ) -> List[TensorValue]:
        input_row_offsets = kwargs["input_row_offsets"]
        assert isinstance(input_row_offsets, TensorValue)
        input_row_offsets_ = distribute_value(input_row_offsets, self.devices)

        return list(
            ops.allreduce.sum(
                [
                    self.list_of_attentions[i](
                        x[i],
                        kv_collections[i],
                        input_row_offsets=input_row_offsets_[i],
                    )
                    for i in range(len(self.devices))
                ]
            )
        )


@dataclass
class AttentionWithRopeQKV(AttentionImplQKV):
    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __call__(
        self,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        wqkv = ops.concat((self.wq, self.wk, self.wv))

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=ops.constant(self.layer_idx, DType.uint32),
            n_heads=self.n_heads,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Cast freqs_cis to xq's dtype to match the fused_qk_ragged_rope kernel.
        freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            ops.constant(self.layer_idx, DType.uint32),
            interleaved=self.rope.interleaved,
        )

        # Calculate Flash Attention.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=ops.constant(self.layer_idx, DType.uint32),
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.wo(attn_out)
