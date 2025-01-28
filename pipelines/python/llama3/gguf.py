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
"""Build a Llama3 model via Graph API from GGUF weights."""

from typing import List, Optional, Union

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorValue, TensorValueLike, ops
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import Weights
from max.pipelines import PipelineConfig, RopeType, WeightsFormat
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from nn import (
    MLP,
    AttentionWithRope,
    DistributedAttentionWithRope,
    DistributedMLP,
    DistributedRMSNorm,
    DistributedTransformer,
    DistributedTransformerBlock,
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


def distribute_value(
    v: TensorValue, devices: List[DeviceRef]
) -> List[TensorValue]:
    return [v.to(device) for device in devices]


def shard_col_value(
    x: TensorValueLike, devices: List[DeviceRef]
) -> List[TensorValue]:
    n_devices = len(devices)
    v = TensorValue(x)
    col_size = v.shape[1].dim // n_devices
    return [
        v[:, i * col_size : (i + 1) * col_size].to(device)
        for i, device in enumerate(devices)
    ]


def shard_row_value(
    x: TensorValueLike, devices: List[DeviceRef]
) -> List[TensorValue]:
    n_devices = len(devices)
    v = TensorValue(x)
    row_size = v.shape[0].dim // n_devices
    return [
        v[i * row_size : (i + 1) * row_size, :].to(device)
        for i, device in enumerate(devices)
    ]


def feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
) -> MLP:
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


def distributed_rms_norm(
    dims: int,
    eps: float,
    weights: Weights,
    devices: List[DeviceRef],
) -> DistributedRMSNorm:
    weights_ = TensorValue(weights.weight.allocate(DType.float32, [dims]))
    weights_devs = distribute_value(weights_, devices)

    rms_norms = [RMSNorm(weights_dev, eps) for weights_dev in weights_devs]

    return DistributedRMSNorm(rms_norms, devices)


def embedding(
    pipeline_config: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: Weights,
) -> Embedding:
    return Embedding(
        weights.weight.allocate(
            pipeline_config.dtype,
            [vocab_size, hidden_dim],
            pipeline_config.quantization_encoding.quantization_encoding,
        )
    )


def distributed_feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
    devices: List[DeviceRef],
) -> DistributedMLP:
    w_ffn_down_full = weights.ffn_down.weight.allocate(
        dtype, [hidden_dim, feed_forward_length], quantization_encoding
    )
    ffn_down_sharded = shard_col_value(w_ffn_down_full, devices)
    w_ffn_gate_full = weights.ffn_gate.weight.allocate(
        dtype, [feed_forward_length, hidden_dim], quantization_encoding
    )
    ffn_gate_sharded = shard_row_value(w_ffn_gate_full, devices)
    w_ffn_up_full = weights.ffn_up.weight.allocate(
        dtype, [feed_forward_length, hidden_dim], quantization_encoding
    )
    ffn_up_sharded = shard_row_value(w_ffn_up_full, devices)

    mlps = [
        MLP(
            Linear(ffn_gate_sharded[rank]),
            Linear(ffn_down_sharded[rank]),
            Linear(ffn_up_sharded[rank]),
        )
        for rank in range(len(devices))
    ]

    return DistributedMLP(mlps, len(devices))


def distributed_attention_opaque(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: Weights,
    layer_idx: TensorValue,
    devices: List[DeviceRef],
) -> DistributedAttentionWithRope:
    wq_full = ops.transpose(
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
    wk_full = ops.transpose(
        weights.attn_k.weight.allocate(
            pipeline_config.dtype,
            [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
            pipeline_config.quantization_encoding.quantization_encoding,
        ),
        0,
        1,
    )
    wv_full = ops.transpose(
        weights.attn_v.weight.allocate(
            pipeline_config.dtype,
            [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
            pipeline_config.quantization_encoding.quantization_encoding,
        ),
        0,
        1,
    )

    wo_full = weights.attn_output.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
        ],
        pipeline_config.quantization_encoding.quantization_encoding,
    )
    wq_shards = shard_col_value(wq_full, devices)
    wk_shards = shard_col_value(wk_full, devices)
    wv_shards = shard_col_value(wv_full, devices)

    # Didn't transpose here since linear will transpose so shard on col instead
    # of row
    wo_shards = shard_col_value(wo_full, devices)
    attns = [
        AttentionWithRope(
            n_heads=pipeline_config.huggingface_config.num_attention_heads
            // len(devices),
            kv_params=kv_params,
            wqkv=ops.concat(
                (wq_shards[rank], wk_shards[rank], wv_shards[rank]), axis=1
            ).transpose(0, 1),
            wo=Linear(wo_shards[rank]),
            rope=rope,
            layer_idx=layer_idx,
        )
        for rank in range(len(devices))
    ]

    return DistributedAttentionWithRope(attns, devices)


def _attention_opaque(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: Weights,
    layer_idx: TensorValue,
) -> AttentionWithRope:
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


def distributed_transformer_opaque(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    max_seq_len: int,
    kv_params: KVCacheParams,
) -> DistributedTransformer:
    devices = [
        DeviceRef(spec.device_type, spec.id)
        for spec in pipeline_config.device_specs
    ]
    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        interleaved_rope_weights = (
            pipeline_config.weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        rope = OptimizedRotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        layers = [
            DistributedTransformerBlock(
                attention=distributed_attention_opaque(
                    kv_params,
                    pipeline_config,
                    rope,
                    weights.blk[i],
                    layer_idx=ops.constant(i, DType.uint32),
                    devices=devices,
                ),
                mlp=distributed_feed_forward(  # type: ignore
                    pipeline_config.dtype,
                    pipeline_config.quantization_encoding.quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.blk[i],
                    devices=devices,
                ),
                attention_norm=distributed_rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.blk[i].attn_norm,
                    devices=devices,
                ),
                mlp_norm=distributed_rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.blk[i].ffn_norm,
                    devices=devices,
                ),
                devices=devices,
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
        else:
            raise ValueError(
                "Unsupported caching strategy " + str(kv_params.cache_strategy)
            )

        return DistributedTransformer(
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
            devices=devices,
            all_logits=pipeline_config.enable_echo,
        )


def _transformer_opaque(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    max_seq_len: int,
    kv_params: KVCacheParams,
) -> Transformer:
    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        interleaved_rope_weights = (
            pipeline_config.weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        rope = OptimizedRotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        if pipeline_config.huggingface_config.model_type == "exaone":
            rms_norm_eps = pipeline_config.huggingface_config.layer_norm_epsilon
        else:
            rms_norm_eps = pipeline_config.huggingface_config.rms_norm_eps
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
                    rms_norm_eps,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    rms_norm_eps,
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
                rms_norm_eps,
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
) -> NaiveAttentionWithRope:
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
    max_seq_len: int,
    kv_params: KVCacheParams,
) -> Union[Transformer, NaiveTransformer]:
    if pipeline_config.cache_strategy.uses_opaque():
        return _transformer_opaque(
            graph=graph,
            pipeline_config=pipeline_config,
            weights=weights,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )

    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        interleaved_rope_weights = (
            pipeline_config.weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        rope = RotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        if pipeline_config.huggingface_config.model_type == "exaone":
            rms_norm_eps = pipeline_config.huggingface_config.layer_norm_epsilon
        else:
            rms_norm_eps = pipeline_config.huggingface_config.rms_norm_eps
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
                    rms_norm_eps,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    rms_norm_eps,
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
                rms_norm_eps,
                weights.output_norm,
            ),
            output=output,
            theta=pipeline_config.huggingface_config.rope_theta,
            embedding=embedding_layer,
        )
