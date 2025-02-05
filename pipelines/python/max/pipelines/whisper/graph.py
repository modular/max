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

from max.dtype import DType
from max.graph import ops
from max.graph.weights import SafetensorWeights
from max.pipelines import PipelineConfig
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
)
from max.pipelines.nn import (
    Conv1D,
    Embedding,
    Linear,
    LPLayerNorm,
    Sequential,
    TransformerBlock,
)

from .encoder import WhisperEncoder


def conv1d(
    dtype: DType,
    in_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    out_channels: int,
    weights: SafetensorWeights,
) -> Conv1D:
    """Creates a 1D convolution layer.
    For conv1: ( hugging_face weights: model.encoder.conv1.weight)
    in_channels = 128
    out_channels = 1280
    kernel_size = 3
    stride = 1
    padding = 1

    For conv2: ( hugging_face weights: model.encoder.conv2.weight)
    in_channels = 1280
    out_channels = 1280
    kernel_size = 3
    stride = 2
    padding = 1
    """
    # Loaded torch weights shape = (out_channels, in_channels, kernel_size) = [1280, 128, 3].
    # Graph-API Conv1D expects (kernel_size, in_channels, out_channels) = [3, 128, 1280].
    # TODO: Implement Conv1D with bias and use it here.
    bias = weights.bias.allocate(dtype, [out_channels])
    return Conv1D(
        filter=ops.permute(
            weights.weight.allocate(
                dtype, [out_channels, in_channels, 1, kernel_size], None
            ),
            [2, 1, 0],
        ),
        stride=stride,
        padding=padding,
    )


def embedding(
    dtype: DType,
    max_source_positions: int,
    hidden_dim: int,
    weights: SafetensorWeights,
):
    return Embedding(
        weights.weight.allocate(
            dtype,
            [max_source_positions, hidden_dim],
        )
    )


def layer_norm(
    dims: int, eps: float, weights: SafetensorWeights
) -> LPLayerNorm:
    # TODO: check the shape of bias
    return LPLayerNorm(
        weight=weights.weight.allocate(DType.bfloat16, [dims]),
        eps=eps,
        bias=weights.bias.allocate(DType.bfloat16, [dims]),
    )


def linear(
    dtype: DType,
    in_features: int,
    out_features: int,
    weights: SafetensorWeights,
) -> Linear:
    # TODO: Check we are passing the correct dim for bias
    return Linear(
        weights.weight.allocate(dtype, [in_features, out_features], None),
        bias=weights.bias.allocate(dtype, [out_features], None),
    )


def feed_forward(
    dtype: DType,
    hidden_dim: int,
    feed_forward_length: int,
    weights: SafetensorWeights,
):
    return Sequential(
        layers=[
            linear(
                dtype,
                feed_forward_length,
                hidden_dim,
                weights.fc1,
            ),
            ops.gelu,
            linear(
                dtype,
                hidden_dim,
                feed_forward_length,
                weights.fc2,
            ),
        ]
    )


def attention(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
    layer_index: int,
):
    wq = weights.self_attn.q_proj.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.d_model,
            pipeline_config.huggingface_config.d_model,
        ],
    )
    wk = weights.self_attn.k_proj.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.d_model,
            pipeline_config.huggingface_config.d_model,
        ],
    )
    wv = weights.self_attn.v_proj.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.d_model,
            pipeline_config.huggingface_config.d_model,
        ],
    )
    wqkv = ops.concat((wq, wk, wv))

    # TODO: v_proj, q_proj, and out_proj attention projections have biases.
    b_v = weights.self_attn.v_proj.bias.allocate(
        pipeline_config.dtype, [pipeline_config.huggingface_config.d_model]
    )
    b_q = weights.self_attn.q_proj.bias.allocate(
        pipeline_config.dtype, [pipeline_config.huggingface_config.d_model]
    )
    b_o = weights.self_attn.out_proj.bias.allocate(
        pipeline_config.dtype, [pipeline_config.huggingface_config.d_model]
    )

    # TODO: Implement AttentionWithoutMask with Bias and use it here.


def encoder(
    pipeline_config: PipelineConfig,
    weights: SafetensorWeights,
    kv_params: KVCacheParams,
) -> WhisperEncoder:
    conv1 = conv1d(
        dtype=pipeline_config.dtype,
        in_channels=pipeline_config.huggingface_config.num_mel_bins,
        kernel_size=3,
        stride=1,
        padding=1,
        out_channels=pipeline_config.huggingface_config.d_model,
        weights=weights.model.encoder.conv1,
    )

    conv2 = conv1d(
        dtype=pipeline_config.dtype,
        in_channels=pipeline_config.huggingface_config.d_model,
        kernel_size=3,
        stride=2,
        padding=1,
        out_channels=pipeline_config.huggingface_config.d_model,
        weights=weights.model.encoder.conv2,
    )

    # TODO: Not sure how to handle this. It learns embeddings to a max size.
    embed_positions = embedding(
        dtype=pipeline_config.dtype,
        max_source_positions=pipeline_config.huggingface_config.max_source_positions,
        hidden_dim=pipeline_config.huggingface_config.d_model,
        weights=weights.model.encoder.embed_positions,
    )

    # EncoderBlocks
    # TODO: Which cache strategy to use? Will both Continuous and paged will work?
    layers = [
        TransformerBlock(
            attention=attention(
                pipeline_config,
                weights.language_model.model.layers[i],
                kv_params,
                layer_idx=ops.constant(i, DType.uint32),  # type: ignore
            ),
            mlp=feed_forward(
                pipeline_config.dtype,
                pipeline_config.huggingface_config.d_model,
                pipeline_config.huggingface_config.encoder_ffn_dim,
                weights.model.encoder.layers[i],
            ),
            attention_norm=layer_norm(
                dims=pipeline_config.huggingface_config.d_model,
                eps=1e-5,
                weights=weights.model.encoder.layers[i].self_attn_layer_norm,
            ),
            mlp_norm=layer_norm(
                dims=pipeline_config.huggingface_config.d_model,
                eps=1e-5,
                weights=weights.model.encoder.layers[i].final_layer_norm,
            ),
        )
        for i in range(pipeline_config.huggingface_config.encoder_layers)
    ]

    # Hugging Face model uses default eps for nn.LayerNorm which is = 1e-5
    norm = layer_norm(
        dims=pipeline_config.huggingface_config.d_model,
        eps=1e-5,
        weights=weights.model.encoder.layer_norm,
    )

    return WhisperEncoder(
        conv1=conv1,
        conv2=conv2,
        embed_positions=embed_positions,
        layers=layers,
        norm=norm,
        kv_params=kv_params,
        kv_collection_constructor=FetchContinuousBatchingKVCacheCollection(
            kv_params
        ),
        all_logits=False,
    )
