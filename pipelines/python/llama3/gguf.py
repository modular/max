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
"""Build a Llama3 model via Graph API from GGUF weights."""

from typing import Optional, Union

from max.dtype import DType
from max.graph import Graph, ops
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import Weights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    Linear,
    NaiveAttentionWithRope,
    NaiveTransformer,
    NaiveTransformerBlock,
    OptimizedRotaryEmbedding,
    RMSNorm,
    RotaryEmbedding,
    Transformer,
    TransformerBlock,
)


def feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
):
    return MLP(
        linear(
            dtype,
            quantization_encoding,
            feed_forward_length,
            hidden_dim,
            weights.ffn_gate,
        ),
        linear(
            dtype,
            quantization_encoding,
            hidden_dim,
            feed_forward_length,
            weights.ffn_down,
        ),
        linear(
            dtype,
            quantization_encoding,
            feed_forward_length,
            hidden_dim,
            weights.ffn_up,
        ),
    )


def linear(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    in_features: int,
    out_features: int,
    weights: Weights,
) -> Linear:
    return Linear(
        weights.weight.allocate(
            dtype, [in_features, out_features], quantization_encoding
        )
    )


def rms_norm(dims: int, eps: float, weights: Weights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.float32, [dims]), eps)


def embedding(
    pipeline_config: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: Weights,
):
    return Embedding(
        weights.weight.allocate(
            pipeline_config.dtype,
            [vocab_size, hidden_dim],
            pipeline_config.quantization_encoding.quantization_encoding,
        )
    )


def _attention_opaque(
    kv_params,
    pipeline_config: PipelineConfig,
    rope,
    weights,
    layer_idx,
):
    wq = ops.transpose(
        weights.attn_q.weight.allocate(
            pipeline_config.dtype,
            [
                pipeline_config.huggingface_config.hidden_size,
                pipeline_config.huggingface_config.hidden_size,
            ],
            pipeline_config.quantization_encoding.quantization_encoding,
        ),
        0,
        1,
    )
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads
    wk = ops.transpose(
        weights.attn_k.weight.allocate(
            pipeline_config.dtype,
            [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
            pipeline_config.quantization_encoding.quantization_encoding,
        ),
        0,
        1,
    )
    wv = ops.transpose(
        weights.attn_v.weight.allocate(
            pipeline_config.dtype,
            [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
            pipeline_config.quantization_encoding.quantization_encoding,
        ),
        0,
        1,
    )

    wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)

    return AttentionWithRope(
        n_heads=pipeline_config.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_output,
        ),
        rope=rope,
        layer_idx=layer_idx,
    )


def _transformer_opaque(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    kv_params: KVCacheParams,
):
    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        rope = OptimizedRotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=pipeline_config.huggingface_config.max_seq_len,
            rope_scaling=rope_scaling,
        )

        layers = [
            TransformerBlock(
                attention=_attention_opaque(
                    kv_params,
                    pipeline_config,
                    rope,
                    weights.blk[i],
                    layer_idx=ops.constant(i, DType.uint32),
                ),
                mlp=feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.quantization_encoding.quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.blk[i],
                ),
                attention_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.blk[i].ffn_norm,
                ),
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            pipeline_config,
            pipeline_config.huggingface_config.vocab_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.token_embd,
        )

        # Smaller model variants lack dedicated weights for a final linear
        # layer, and share the embedding layer.
        if weights.output.weight.exists():
            output = linear(
                pipeline_config.dtype,
                pipeline_config.quantization_encoding.quantization_encoding,
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
                weights.output,
            )
        else:
            output = Linear(embedding_layer.weights)

        if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy " + str(kv_params.cache_strategy)
            )

        return Transformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                pipeline_config.huggingface_config.hidden_size,
                pipeline_config.huggingface_config.rms_norm_eps,
                weights.output_norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
            all_logits=pipeline_config.enable_echo,
        )


def attention(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: Union[OptimizedRotaryEmbedding, RotaryEmbedding],
    weights: Weights,
):
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads
    return NaiveAttentionWithRope(
        n_heads=pipeline_config.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        dim=pipeline_config.huggingface_config.hidden_size,
        wk=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            kv_weight_dim,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_k,
        ),
        wv=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            kv_weight_dim,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_v,
        ),
        wq=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_q,
        ),
        wo=linear(
            pipeline_config.dtype,
            pipeline_config.quantization_encoding.quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_output,
        ),
        rope=rope,
    )


def transformer(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    kv_params: KVCacheParams,
):
    if pipeline_config.cache_strategy.uses_opaque():
        return _transformer_opaque(graph, pipeline_config, weights, kv_params)

    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        rope = RotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=pipeline_config.huggingface_config.max_seq_len,
            rope_scaling=rope_scaling,
        )

        layers = [
            NaiveTransformerBlock(
                attention=attention(
                    kv_params, pipeline_config, rope, weights.blk[i]
                ),
                mlp=feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.quantization_encoding.quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.blk[i],
                ),
                attention_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.blk[i].ffn_norm,
                ),
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            pipeline_config,
            pipeline_config.huggingface_config.vocab_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.token_embd,
        )

        # Smaller model variants lack dedicated weights for a final linear
        # layer, and share the embedding layer.
        if weights.output.weight.exists():
            output = linear(
                pipeline_config.dtype,
                pipeline_config.quantization_encoding.quantization_encoding,
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
                weights.output,
            )
        else:
            output = Linear(embedding_layer.weights)

        return NaiveTransformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                pipeline_config.huggingface_config.hidden_size,
                pipeline_config.huggingface_config.rms_norm_eps,
                weights.output_norm,
            ),
            output=output,
            theta=pipeline_config.huggingface_config.rope_theta,
            embedding=embedding_layer,
        )
